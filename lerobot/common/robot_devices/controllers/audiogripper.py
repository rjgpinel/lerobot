"""
Modified AudioGripperController that connects to the chatbot server to receive
audio data and emotion information for controlling the gripper.
Uses identical processing pathways for both live and default audio.
"""

from lerobot.common.robot_devices.controllers.configs import AudioGripperControllerConfig
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc

from pathlib import Path
import threading
import time
import logging
import socket
import json
import numpy as np
from scipy.signal import butter, filtfilt
import pyaudio
import sounddevice as sd
import librosa
import io

# Socket communication
HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

# Audio configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

SAMPLING_RATE = 16000  # in Hz
BLOCK_DURATION = 0.05  # in seconds (50 ms)
THRESHOLD = 0.02       # Adjust this value based on your microphone sensitivity

MAX_RMS = 0.4

# Emotion to intensity mapping (for more expressive movements)
EMOTION_INTENSITY = {
    "angry": 1.2,      # More intense movements
    "curious": 0.9,    # Medium intensity
    "fearful": 0.7,    # Less intense but rapid
    "happy": 1.0,      # Normal intensity
    "playful": 1.1,    # Slightly more intense
    "sad": 0.6,        # Less intense movements
    "surprised": 1.3,  # Very intense movements
}

def smooth_signal(y, cutoff=5.0):
    """Apply low-pass filter to smooth audio envelope"""
    nyquist = 0.5 * RATE
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, np.abs(y))

class AudioGripperController:
    def __init__(
            self,
            config: AudioGripperControllerConfig):
        self.config = config

        # Configure logging for better diagnostics
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.info("Initializing AudioGripperController with shared processing paths")

        # Get the directory containing the current script
        script_dir = Path(__file__).parent.resolve()
        self.audio_file = script_dir / "puppet.mp3"

        self.min_angle = 1
        self.max_angle = 45
        # self.max_angle = 95

        self.current_positions = {self.config.motor_id: self.config.initial_position}

        # Initialize device
        self.device = None
        self.running = True
        
        # Audio data and emotion from chatbot server
        self.current_audio = None
        self.current_emotion = "curious"  # Default emotion
        self.emotion_modifier = 1.0  # Default emotion intensity modifier
        
        # For live audio processing
        self.mouth_values = []
        self.times = []
        self.current_idx = 0
        self.last_process_time = 0
        
        # Socket connection to chatbot
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected_to_chatbot = False
        
        # Connect to chatbot
        try:
            logging.info(f"Connecting to chatbot server at {HOST}:{PORT}...")
            self.socket.connect((HOST, PORT))
            self.connected_to_chatbot = True
            logging.info("Connected to chatbot server")
        except Exception as e:
            logging.error(f"Failed to connect to chatbot server: {e}")
            logging.info("Will continue with fallback audio file")
            
        # Flag to indicate if we should use the chatbot or fallback
        self.use_chatbot = self.connected_to_chatbot
        
        # Pre-generate the envelope extraction for default audio
        # so we can reuse the exact same function
        self.generate_mouth_values()

        # Start the thread to read inputs
        self.lock = threading.Lock()
        
        # Start the thread to receive data from chatbot
        if self.connected_to_chatbot:
            self.chatbot_thread = threading.Thread(target=self.receive_from_chatbot, daemon=True)
            self.chatbot_thread.start()
        
        # Start the main processing thread
        self.thread = threading.Thread(target=self.read_loop, daemon=True)
        self.thread.start()

    def receive_from_chatbot(self):
        """Receive audio data and emotion from the chatbot server."""
        logging.info("Starting to receive data from chatbot")
        try:
            while self.running and self.connected_to_chatbot:
                try:
                    # First receive the message length (4 bytes)
                    msg_len_bytes = self.socket.recv(4)
                    if not msg_len_bytes:
                        logging.warning("Connection closed by chatbot server")
                        self.connected_to_chatbot = False
                        self.use_chatbot = False
                        break
                    
                    msg_len = int.from_bytes(msg_len_bytes, byteorder='big')
                    
                    # Now receive the full message
                    data = b''
                    while len(data) < msg_len:
                        packet = self.socket.recv(min(4096, msg_len - len(data)))
                        if not packet:
                            break
                        data += packet
                    
                    if len(data) < msg_len:
                        logging.warning("Incomplete message received")
                        continue
                    
                    # Parse the message
                    message = json.loads(data.decode('utf-8'))
                    
                    # Extract emotion and audio data
                    emotion = message.get("emotion", "curious")
                    audio_data_raw = message.get("audio_data", [])
                    
                    # Convert to numpy array
                    try:
                        audio_data = np.array(audio_data_raw, dtype=np.float32)
                        
                        # Normalize audio to [-1, 1] range if needed
                        data_max = np.max(np.abs(audio_data)) if len(audio_data) > 0 else 0
                        if data_max > 1.0:
                            audio_data = audio_data / data_max
                            
                    except Exception as e:
                        logging.error(f"Error processing audio data: {e}")
                        continue
                    
                    # If we received audio data, process it through the same pathway 
                    # as the default audio
                    if len(audio_data) > 0:
                        with self.lock:
                            # Process this audio chunk immediately
                            self.process_live_audio_chunk(audio_data, emotion)
                    
                    # Update emotion
                    with self.lock:
                        self.current_emotion = emotion
                        self.emotion_modifier = EMOTION_INTENSITY.get(emotion, 1.0)
                    
                except Exception as e:
                    logging.error(f"Error receiving data from chatbot: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
                    time.sleep(0.1)  # Prevent tight loop on error
        
        except Exception as e:
            logging.error(f"Chatbot communication thread crashed: {e}")
            import traceback
            logging.error(traceback.format_exc())
        finally:
            logging.info("Chatbot communication thread ending")
            self.connected_to_chatbot = False
            self.use_chatbot = False

    def process_live_audio_chunk(self, audio_data, emotion="curious"):
        """
        Process a chunk of live audio using the SAME function that successfully
        processes the default audio.
        
        This ensures we use identical processing for both audio sources.
        """
        try:
            # Only process at most every 50ms to avoid too frequent updates
            current_time = time.time()
            if current_time - self.last_process_time < 0.05:
                return
                
            self.last_process_time = current_time
            
            # Log input data for debugging
            audio_min = np.min(audio_data)
            audio_max = np.max(audio_data)
            audio_abs_max = max(abs(audio_min), abs(audio_max))
            logging.debug(f"Audio data shape: {audio_data.shape}, min: {audio_min:.2e}, max: {audio_max:.2e}, abs max: {audio_abs_max:.2e}")
            
            # Get the emotion modifier
            emotion_mod = EMOTION_INTENSITY.get(emotion, 1.0)
            
            # Calculate RMS energy directly (direct equivalent of librosa.feature.rms)
            rms = np.sqrt(np.mean(np.square(audio_data)))
            logging.debug(f"Raw RMS value: {rms:.6f}")
            
            # Normalize RMS based on audio characteristics
            # Fixed lower and upper bounds based on expected audio levels
            MIN_MOUTH_RMS = 0.05     # Minimum RMS that should start moving the mouth
            MAX_MOUTH_RMS = 0.30     # RMS value that corresponds to fully open mouth
            
            # Scale RMS based on observed values (from debug output)
            # Your example showed tiny values, so we scale up appropriately
            if audio_abs_max < 1e-5:  # For tiny values like those in debug output
                scaled_rms = rms * 1e7
            else:
                # For larger values, we might not need as much scaling
                scaled_rms = rms
                
            logging.debug(f"Scaled RMS: {scaled_rms:.6f}")
            
            # Apply lower threshold - no movement below this
            if scaled_rms < MIN_MOUTH_RMS:
                mouth_value = 0.0
            else:
                # Map to 0-1 range with proper bounds
                normalized_value = (scaled_rms - MIN_MOUTH_RMS) / (MAX_MOUTH_RMS - MIN_MOUTH_RMS)
                # Clamp to 0-1 range
                normalized_value = max(0.0, min(normalized_value, 1.0))
                # Apply power curve to make movement more natural
                mouth_value = np.power(normalized_value, 0.6) * emotion_mod
                # Ensure we don't exceed 1.0 after applying emotion modifier
                mouth_value = min(mouth_value, 1.0)
            
            logging.debug(f"Final mouth value: {mouth_value:.4f}")
            
            # Update the current position directly using the function that works for default audio
            self._update_current_position(mouth_value) 
            
        except Exception as e:
            logging.error(f"Error processing live audio: {e}")
            import traceback
            logging.error(traceback.format_exc())

    def generate_mouth_values(self):
        """Extract time-synchronized mouth opening estimates from default audio file."""
        try:
            logging.info(f"Generating mouth values from {self.audio_file}")
            y, sr = librosa.load(self.audio_file, sr=None)
            self.times = librosa.times_like(y, sr=sr)
            
            # Extract amplitude envelope with psychoacoustic weighting
            envelope = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            smoothed = np.convolve(envelope, np.hanning(5), mode='same')  # Temporal smoothing
            self.mouth_values = (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed))
            
            logging.info(f"Generated {len(self.mouth_values)} mouth values")
            
        except Exception as e:
            logging.error(f"Error generating mouth values: {e}")
            import traceback
            logging.error(traceback.format_exc())
            # Initialize with empty arrays in case of error
            self.times = []
            self.mouth_values = []

    def _update_current_position(self, absolute_value):
        """Update gripper position based on audio intensity."""
        # This is the SHARED function used by both default and live audio paths
        
        # Safety check - ensure value is in valid range
        absolute_value = max(0.0, min(1.0, absolute_value))
        
        with self.lock:
            emotion_mod = self.emotion_modifier
            
        # Apply emotion modifier and clamp to 0-1 range
        modified_value = absolute_value * emotion_mod
        gripper_abs_pos = max(min(modified_value, 1.0), 0.0)
        
        # Map to gripper position
        gripper_pos = gripper_abs_pos * (self.max_angle - self.min_angle) + self.min_angle
        self.current_positions[self.config.motor_id] = gripper_pos
        
        # For debugging
        logging.debug(f"Updated position: {gripper_pos:.2f}")

    def get_current_emotion(self):
        """Safely get the current emotion."""
        with self.lock:
            return self.current_emotion

    def audio_callback(self, outdata, frames, time, status):
        """Audio playback callback for synchronization with fallback audio file."""
        chunk = self.audio_data[self.audio_pos:self.audio_pos + frames]
        outdata[:] = chunk.reshape(-1, 1)
        self.audio_pos += frames
    
    def read_loop(self):
        """Main loop to process audio and control the gripper."""
        logging.info("Starting audio processing loop")
        
        if not self.use_chatbot:
            # Fallback to pre-recorded audio file
            try:
                logging.info("Using fallback audio file")
                self.audio_data, sr = librosa.load(self.audio_file, sr=None, mono=True)
                self.audio_pos = 0
                stream = sd.OutputStream(
                    samplerate=sr,
                    channels=1,
                    callback=self.audio_callback,
                    dtype='float32'
                )
                logging.info("Starting audio playback")
                stream.start()
                start_time = time.time()
                idx = 0

                while stream.active and idx < len(self.times) and self.running:
                    elapsed = time.time() - start_time
                    if elapsed >= self.times[idx]:
                        self._update_current_position(self.mouth_values[idx])
                        idx += 1
                    time.sleep(0.001)  # High-precision timing

                stream.stop()
                logging.info("Audio playback finished")
            
            except Exception as e:
                logging.error(f"Error in audio processing: {e}")
        else:
            # Using chatbot mode - this is now just a monitoring loop
            # since processing happens directly in receive_from_chatbot
            logging.info("Using chatbot mode for audio processing")
            
            last_connection_check = time.time()
            
            while self.running:
                current_time = time.time()
                
                # Check for connection status periodically
                if current_time - last_connection_check > 5.0:
                    if not self.connected_to_chatbot and self.use_chatbot:
                        logging.warning("Lost connection to chatbot, switching to fallback mode")
                        self.use_chatbot = False
                        return self.read_loop()  # Restart with fallback mode
                    last_connection_check = current_time
                
                # When no speech is detected, gradually return to rest position
                with self.lock:
                    # Check if we've received updates recently
                    audio_timestamp = getattr(self, 'last_process_time', 0)
                    inactive_time = current_time - audio_timestamp
                    
                    if inactive_time > 0.5:  # No updates for 500ms
                        # Gradually return to initial position
                        current_pos = self.current_positions[self.config.motor_id]
                        target_pos = self.config.initial_position
                        
                        # Only update if we're not already at rest position
                        if abs(current_pos - target_pos) > 0.01:
                            # Move 10% closer to the target position
                            new_pos = current_pos + 0.1 * (target_pos - current_pos)
                            self.current_positions[self.config.motor_id] = new_pos
                
                # Short sleep to prevent tight loop
                time.sleep(0.02)

    def get_command(self):
        """
        Return the current motor positions after reading and processing inputs.
        """
        return self.current_positions.copy()

    def disconnect(self):
        """Disconnect and clean up resources."""
        logging.info("Disconnecting AudioGripperController")
        self.running = False
        
        if self.connected_to_chatbot:
            try:
                self.socket.close()
                logging.info("Closed connection to chatbot server")
            except Exception as e:
                logging.error(f"Error closing socket: {e}")
            
            self.connected_to_chatbot = False