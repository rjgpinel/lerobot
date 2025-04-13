from lerobot.common.robot_devices.controllers.configs import AudioGripperControllerConfig
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc

from pathlib import Path

import threading
import time
import logging

import numpy as np
from scipy.signal import butter, filtfilt
import pyaudio
from math import sin

import sounddevice as sd

import librosa

# Audio configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

SAMPLING_RATE = 16000  # in Hz
BLOCK_DURATION = 0.05  # in seconds (50 ms)
THRESHOLD = 0.02       # Adjust this value based on your microphone sensitivity

MAX_RMS = 0.4

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

        # Initialize hid device
        self.device = None
        self.running = True

        # # init the audio stream
        # self.p = pyaudio.PyAudio()
        # self.stream = self.p.open(format=FORMAT,
        #     channels=CHANNELS,
        #     rate=RATE,
        #     input=True,
        #     frames_per_buffer=CHUNK)

        # self.read_loop()

        # Start the thread to read inputs
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.read_loop, daemon=True)
        self.thread.start()

    def generate_mouth_values(self):
        """Extract time-synchronized mouth opening estimates"""
        y, sr = librosa.load(self.audio_file, sr=None)
        times = librosa.times_like(y, sr=sr)
        
        # Extract amplitude envelope with psychoacoustic weighting
        envelope = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        smoothed = np.convolve(envelope, np.hanning(5), mode='same')  # Temporal smoothing
        normalized = (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed))
        
        return times, normalized

    def _process_audio_input(self, raw_data):
        audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
        smoothed = smooth_signal(audio)
        # gripper_pos = np.mean(smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed)) * (self.max_angle - self.min_angle) + self.min_angle
        gripper_abs_pos = (sin((time.time()*0.5) * 3.1415) +1) *0.5

        gripper_pos = gripper_abs_pos * (self.max_angle - self.min_angle) + self.min_angle
        self.current_positions[self.config.motor_id] = gripper_pos

    # def audio_callback(self, indata, frames, time, status):
    #     if status:
    #         print(f"Audio callback status: {status}")
    #     # Calculate the root mean square (RMS) of the audio block
    #     rms = np.sqrt(np.mean(indata**2))
    #     # Determine gripper state based on RMS value
    #     # gripper_state = 1 if rms > THRESHOLD else 0
    #     gripper_abs_pos = max(min(rms / MAX_RMS, 1.0), 0.0)

    #     gripper_pos = gripper_abs_pos * (self.max_angle - self.min_angle) + self.min_angle
    #     self.current_positions[self.config.motor_id] = gripper_pos

    def _update_current_position(self, absolute_value):
        gripper_abs_pos = max(min(absolute_value, 1.0), 0.0)
        gripper_pos = gripper_abs_pos * (self.max_angle - self.min_angle) + self.min_angle
        self.current_positions[self.config.motor_id] = gripper_pos

    def audio_callback(self, outdata, frames, time, status):
        """Audio playback callback for synchronization"""
        chunk = self.audio_data[self.audio_pos:self.audio_pos + frames]
        outdata[:] = chunk.reshape(-1, 1)
        self.audio_pos += frames


    def get_command(self):
        """
        Return the current motor positions after reading and processing inputs.
        """
        return self.current_positions.copy()
    
    def read_loop(self):
        try:
            # with sd.InputStream(callback=self.audio_callback,
            #     channels=1,
            #     samplerate=SAMPLING_RATE,
            #     blocksize=int(SAMPLING_RATE * BLOCK_DURATION)):
            #     while self.running:
            #         sd.sleep(100)

            times, mouth_values = self.generate_mouth_values()
            self.audio_data, sr = librosa.load(self.audio_file, sr=None, mono=True)
            self.audio_pos = 0
            stream = sd.OutputStream(
                samplerate=sr,
                channels=1,
                callback=self.audio_callback,
                dtype='float32'
            )
            print("Time(s)\tMouthOpen")
            stream.start()
            start_time = time.time()
            idx = 0

            while stream.active and idx < len(times):
                elapsed = time.time() - start_time
                if elapsed >= times[idx]:
                    
                    # print(f"{times[idx]:.2f}\t{mouth_values[idx]:.2f}")
                    self._update_current_position(mouth_values[idx])
                    idx += 1
                time.sleep(0.001)  # High-precision timing

            stream.stop()
        except KeyboardInterrupt:
            print("\nLip-syncing stopped.")
        # while self.running:
        #     try:
        #         raw_data = self.stream.read(CHUNK, exception_on_overflow=False)
        #         if raw_data:
        #             self._process_audio_input(raw_data)
        #     except Exception as e:
        #         logging.error(f"Error reading from device: {e}")
        #         time.sleep(1)  # Wait before retrying
        #         self.connect()
    

    def disconnect(self):
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            logging.info("Controller disconnected.")
            self.stream = None
            self.p.terminate()