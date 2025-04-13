"""
Modified AudioGripperController that connects to the chatbot server to receive
audio data and emotion information for controlling the gripper.
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
                    print(f"{time.monotonic()}: Received audio")
                    message = json.loads(data.decode('utf-8'))
                    
                    # Extract emotion and audio data
                    emotion = message.get("emotion", "curious")
                    audio_data = np.array(message.get("audio_data", []))
                    print(f"{time.monotonic()}: Emotion: {emotion} | audio: {len(audio_data)}")
                    
                    # Update current data
                    with self.lock:
                        self.current_emotion = emotion
                        self.emotion_modifier = EMOTION_INTENSITY.get(emotion, 1.0)
                        self.current_audio = audio_data
                        
                    # Process audio immediately
                    self.process_audio_chunk()
                        
                except Exception as e:
                    logging.error(f"Error receiving data from chatbot: {e}")
                    time.sleep(0.1)  # Prevent tight loop on error
        
        except Exception as e:
            logging.error(f"Chatbot communication thread crashed: {e}")
        finally:
            logging.info("Chatbot communication thread ending")
            self.connected_to_chatbot = False
            self.use_chatbot = False

    def process_audio_chunk(self):
        """Process current audio chunk to control gripper."""
        try:
            with self.lock:
                audio = self.current_audio
                emotion_mod = self.emotion_modifier
                
            if audio is None or len(audio) == 0:
                return
                
            # Calculate RMS of the audio chunk
            rms = np.sqrt(np.mean(np.square(audio)))
            
            # Apply emotion modifier to the RMS
            modified_rms = rms * emotion_mod
            
            # Normalize and clamp to 0-1 range
            normalized_intensity = max(min(modified_rms / MAX_RMS, 1.0), 0.0)
            
            # Map to gripper position
            gripper_pos = normalized_intensity * (self.max_angle - self.min_angle) + self.min_angle
            
            # Update gripper position
            self.current_positions[self.config.motor_id] = gripper_pos
            
        except Exception as e:
            logging.error(f"Error processing audio chunk: {e}")

    def generate_mouth_values(self):
        """Extract time-synchronized mouth opening estimates from default audio file."""
        y, sr = librosa.load(self.audio_file, sr=None)
        times = librosa.times_like(y, sr=sr)
        
        # Extract amplitude envelope with psychoacoustic weighting
        envelope = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        smoothed = np.convolve(envelope, np.hanning(5), mode='same')  # Temporal smoothing
        normalized = (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed))
        
        return times, normalized

    def _update_current_position(self, absolute_value):
        """Update gripper position based on audio intensity."""
        with self.lock:
            emotion_mod = self.emotion_modifier
            
        # Apply emotion modifier and clamp to 0-1 range
        modified_value = absolute_value * emotion_mod
        gripper_abs_pos = max(min(modified_value, 1.0), 0.0)
        
        # Map to gripper position
        gripper_pos = gripper_abs_pos * (self.max_angle - self.min_angle) + self.min_angle
        self.current_positions[self.config.motor_id] = gripper_pos

    def audio_callback(self, outdata, frames, time, status):
        """Audio playback callback for synchronization with fallback audio file."""
        chunk = self.audio_data[self.audio_pos:self.audio_pos + frames]
        outdata[:] = chunk.reshape(-1, 1)
        self.audio_pos += frames

    def get_command(self):
        """
        Return the current motor positions after reading and processing inputs.
        """
        return self.current_positions.copy()
    
    def read_loop(self):
        """Main loop to process audio and control the gripper."""
        logging.info("Starting audio processing loop")
        
        if not self.use_chatbot:
            # Fallback to pre-recorded audio file
            try:
                logging.info("Using fallback audio file")
                times, mouth_values = self.generate_mouth_values()
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

                while stream.active and idx < len(times) and self.running:
                    elapsed = time.time() - start_time
                    if elapsed >= times[idx]:
                        self._update_current_position(mouth_values[idx])
                        idx += 1
                    time.sleep(0.001)  # High-precision timing

                stream.stop()
                logging.info("Audio playback finished")
            
            except Exception as e:
                logging.error(f"Error in audio processing: {e}")
        else:
            # Using chatbot mode - just keep the thread alive
            logging.info("Using chatbot mode for audio processing")
            while self.running:
                time.sleep(0.1)
                
                # If chatbot connection is lost, switch to fallback mode
                if not self.connected_to_chatbot and self.use_chatbot:
                    logging.warning("Lost connection to chatbot, switching to fallback mode")
                    self.use_chatbot = False
                    return self.read_loop()  # Restart with fallback mode

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