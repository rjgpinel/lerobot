"""
Modified AudioGripperController that connects to the chatbot server to receive
audio data and emotion information for controlling the gripper.
Uses consistent audio processing for both live and fallback audio.
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

# Audio processing parameters
FRAME_LENGTH = 2048    # Frame length for envelope extraction
HOP_LENGTH = 512       # Hop length for envelope extraction
SMOOTHING_WINDOW = 5   # Window size for temporal smoothing

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

def extract_mouth_envelope(audio_data, sr=RATE):
    """
    Extract mouth opening envelope from audio data using the same
    high-quality method used for the fallback audio.
    
    Args:
        audio_data: numpy array of audio samples
        sr: sample rate of the audio
        
    Returns:
        normalized envelope values
    """
    # Check if we have enough data for processing
    if len(audio_data) < FRAME_LENGTH:
        # Pad with zeros if needed
        audio_data = np.pad(audio_data, (0, FRAME_LENGTH - len(audio_data)), 'constant')
    
    # Extract RMS energy (same as in generate_mouth_values)
    envelope = librosa.feature.rms(
        y=audio_data, 
        frame_length=FRAME_LENGTH, 
        hop_length=HOP_LENGTH
    )[0]
    
    # Apply temporal smoothing
    if len(envelope) > 1:
        smoothed = np.convolve(envelope, np.hanning(SMOOTHING_WINDOW), mode='same')
    else:
        smoothed = envelope
    
    # Normalize to 0-1 range
    if np.max(smoothed) > np.min(smoothed):
        normalized = (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed))
    else:
        normalized = np.zeros_like(smoothed)
    
    return normalized

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
        
        # Buffer for recent audio (to have enough context for processing)
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_max_size = RATE * 1  # 1 second of audio at RATE
        
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
                    message = json.loads(data.decode('utf-8'))
                    
                    # Extract emotion and audio data
                    emotion = message.get("emotion", "curious")
                    audio_data = np.array(message.get("audio_data", []))
                    
                    # Update current data with timestamp
                    with self.lock:
                        self.current_emotion = emotion
                        self.emotion_modifier = EMOTION_INTENSITY.get(emotion, 1.0)
                        self.current_audio = audio_data
                        
                        # Add new audio to buffer
                        self.update_audio_buffer(audio_data)
                        
                        # Add timestamp for when audio was last updated
                        self.audio_timestamp = time.time()
                        
                    logging.debug(f"Received audio chunk: {len(audio_data)} samples, emotion: {emotion}")
                        
                except Exception as e:
                    logging.error(f"Error receiving data from chatbot: {e}")
                    time.sleep(0.1)  # Prevent tight loop on error
        
        except Exception as e:
            logging.error(f"Chatbot communication thread crashed: {e}")
        finally:
            logging.info("Chatbot communication thread ending")
            self.connected_to_chatbot = False
            self.use_chatbot = False

    def update_audio_buffer(self, new_audio):
        """Update the audio buffer with new audio data."""
        if len(new_audio) == 0:
            return
            
        # Append new audio to buffer
        self.audio_buffer = np.append(self.audio_buffer, new_audio)
        
        # Trim buffer if it's too long
        if len(self.audio_buffer) > self.buffer_max_size:
            self.audio_buffer = self.audio_buffer[-self.buffer_max_size:]

    def process_audio_chunk(self):
        """Process current audio buffer to control gripper using the same method as fallback audio."""
        try:
            with self.lock:
                audio_buffer = self.audio_buffer.copy()
                emotion_mod = self.emotion_modifier
                
            if len(audio_buffer) == 0:
                return
                
            # Extract mouth envelope using the same method as the fallback audio
            normalized_envelope = extract_mouth_envelope(audio_buffer)
            
            # Use the last value from the envelope as the current mouth position
            if len(normalized_envelope) > 0:
                mouth_position = normalized_envelope[-1]
                
                # Apply emotion modifier
                modified_position = mouth_position * emotion_mod
                
                # Map to gripper position
                gripper_pos = modified_position * (self.max_angle - self.min_angle) + self.min_angle
                
                # Update gripper position
                self.current_positions[self.config.motor_id] = gripper_pos
                
                # Debug logging
                if logging.getLogger().level <= logging.DEBUG:
                    logging.debug(f"Mouth position: {mouth_position:.4f}, Modified: {modified_position:.4f}, Gripper: {gripper_pos:.2f}")
            
        except Exception as e:
            logging.error(f"Error processing audio chunk: {e}")

    def generate_mouth_values(self):
        """Extract time-synchronized mouth opening estimates from default audio file."""
        y, sr = librosa.load(self.audio_file, sr=None)
        times = librosa.times_like(y, sr=sr)
        
        # Extract amplitude envelope with psychoacoustic weighting
        envelope = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        smoothed = np.convolve(envelope, np.hanning(SMOOTHING_WINDOW), mode='same')  # Temporal smoothing
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

    def get_current_emotion(self):
        """Safely get the current emotion.
        
        Returns:
            str: The current emotion
        """
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
            # Using chatbot mode - actively process received audio data
            logging.info("Using chatbot mode for audio processing")
            last_processed_time = time.time()
            last_audio_update_time = time.time()
            is_speaking = False
            
            while self.running:
                current_time = time.time()
                
                # Check if we have fresh audio data to process
                with self.lock:
                    has_audio = len(self.audio_buffer) > 0
                    # Get a timestamp for when audio was last updated
                    audio_timestamp = getattr(self, 'audio_timestamp', 0)
                
                # Check if audio was recently updated (within last 1 second)
                fresh_audio = (current_time - audio_timestamp) < 1.0
                
                if has_audio and fresh_audio:
                    # We have fresh audio data - consider this active speech
                    is_speaking = True
                    last_audio_update_time = current_time
                    
                    # Process audio at a reasonable rate (50Hz)
                    if current_time - last_processed_time > 0.02:
                        self.process_audio_chunk()
                        last_processed_time = current_time
                
                elif has_audio and is_speaking:
                    # We have audio but it's not fresh - continue processing for a short while
                    # This handles small gaps in speech
                    if current_time - last_processed_time > 0.02:
                        self.process_audio_chunk()
                        last_processed_time = current_time
                    
                    # Check if we should stop speaking mode
                    if current_time - last_audio_update_time > 0.5:  # 500ms with no new audio
                        is_speaking = False
                        logging.debug("Speech ended, returning to rest position")
                
                else:
                    # No active speech, return to rest position
                    if not is_speaking:
                        # Gradually return to initial position
                        current_pos = self.current_positions[self.config.motor_id]
                        target_pos = self.config.initial_position
                        
                        # Only update if we're not already at rest position
                        if abs(current_pos - target_pos) > 0.01:
                            # Move 20% closer to the target position (faster return)
                            new_pos = current_pos + 0.2 * (target_pos - current_pos)
                            self.current_positions[self.config.motor_id] = new_pos
                
                # Check if chatbot connection is lost
                if not self.connected_to_chatbot and self.use_chatbot:
                    logging.warning("Lost connection to chatbot, switching to fallback mode")
                    self.use_chatbot = False
                    return self.read_loop()  # Restart with fallback mode
                
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