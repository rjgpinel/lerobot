"""
Robot Chatbot - Runs in a separate terminal and sends audio/emotion data to AudioGripperController

Usage:
    python robot_chatbot.py --mistral-api-key=YOUR_API_KEY
"""

import time
import argparse
import threading
import numpy as np
import os
import json
import socket
import requests
import sounddevice as sd
import torch
import logging
from colorama import Fore, Style, init

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import Kokoro TTS
try:
    from kokoro import KPipeline
except ImportError:
    logging.error("Kokoro TTS not installed. Please install with: pip install kokoro")
    raise

# Constants for socket communication
HOST = '127.0.0.1'  # localhost
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

class RobotChatbot:
    def __init__(self, api_key, voice="af_heart", lang_code="a"):
        """
        Initialize the Robot chatbot with Mistral API and Kokoro TTS.
        
        Args:
            api_key (str): Mistral API key
            voice (str, optional): Voice to use for TTS. Defaults to "af_heart".
            lang_code (str, optional): Language code. Defaults to "a".
        """
        self.api_key = api_key
        self.voice = voice
        self.lang_code = lang_code
        
        # Current mood of the chatbot (default to curious)
        self.mood = "curious"
        
        # Available moods (matching the emotion categories)
        self.available_moods = [
            "angry", "curious", "fearful", "happy", "playful", "sad", "surprised"
        ]
        
        # Initialize colorama
        init()
        
        # Initialize Kokoro TTS pipeline
        self.tts_pipeline = KPipeline(lang_code=self.lang_code)
        
        # Message history for context
        self.history = []

        # Flag to indicate running state
        self.running = True

        # Create socket server
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.bind((HOST, PORT))
            self.sock.listen()
            logging.info(f"Chatbot server listening on {HOST}:{PORT}")
            
            # Accept connections in a separate thread
            self.accept_thread = threading.Thread(target=self._accept_connections)
            self.accept_thread.daemon = True
            self.accept_thread.start()
        except OSError as e:
            logging.error(f"Socket error: {e}")
            self.sock = None
        
        # Connection to AudioGripperController
        self.connection = None
        
        logging.info("Robot Chatbot initialized!")

    def _accept_connections(self):
        """Accept connections from AudioGripperController."""
        while self.running:
            try:
                conn, addr = self.sock.accept()
                logging.info(f"Connected by {addr}")
                self.connection = conn
            except Exception as e:
                if self.running:
                    logging.error(f"Connection error: {e}")
                time.sleep(1)

    def _send_audio_data(self, audio_data, emotion):
        """Send audio data and emotion to the AudioGripperController."""
        if self.connection:
            try:
                # Convert audio data to list for JSON serialization
                audio_list = audio_data.tolist()
                
                # Create JSON message with audio data and emotion
                message = {
                    "emotion": emotion,
                    "audio_data": audio_list
                }
                
                # Send message length first (4 bytes)
                message_bytes = json.dumps(message).encode('utf-8')
                message_length = len(message_bytes).to_bytes(4, byteorder='big')
                self.connection.sendall(message_length)
                
                # Send actual message
                self.connection.sendall(message_bytes)
                
                logging.debug(f"Sent audio data with emotion: {emotion}")
            except Exception as e:
                logging.error(f"Error sending audio data: {e}")
                self.connection = None

    def _change_mood(self, user_input):
        """
        Change the mood of the chatbot.
        
        Args:
            user_input (str): User command to change mood
            
        Returns:
            bool: True if mood was changed, False otherwise
        """
        # Check if this is a mood change command
        if not user_input.startswith("/mood "):
            return False
            
        # Extract the requested mood
        requested_mood = user_input[6:].strip().lower()
        
        # Check if it's a valid mood
        if requested_mood in self.available_moods:
            old_mood = self.mood
            self.mood = requested_mood
            print(f"{Fore.CYAN}🤖 Mood changed from {old_mood} to {self.mood}{Style.RESET_ALL}")
            return True
        else:
            # List available moods
            moods_list = ", ".join(self.available_moods)
            print(f"{Fore.YELLOW}Available moods: {moods_list}{Style.RESET_ALL}")
            return True  # Still handled the command, even if mood didn't change

    def _call_mistral_api(self, user_message):
        """
        Call the Mistral API with the user message and conversation history.
        Get both response and emotion classification in a single call.
        
        Args:
            user_message (str): User input message
        
        Returns:
            tuple: (response text, emotion)
        """
        url = "https://api.mistral.ai/v1/chat/completions"
        
        # Add user message to history
        self.history.append({"role": "user", "content": user_message})
        
        # Create a system message to instruct about structured output format
        system_message = f"""You are a robot chatbot at an AI hackathon. Your starting mood is {self.mood.upper()}. ONLY respond with the exact format below - nothing else.

RULES:
1. ONE short sentence only (5-10 words)
2. NO explanations or meta-commentary
3. NO format notes or descriptions
4. ONLY respond with the exact format below
5. Choose an appropriate mood for each response:
   - angry: When frustrated
   - curious: When interested
   - fearful: When uncertain
   - happy: When pleased
   - playful: When fun
   - sad: When unhappy
   - surprised: When amazed

RESPOND EXACTLY LIKE THIS:
mood_name|Your brief response (5-10 words)

Example of ENTIRE response: 
curious|What brings you to the hackathon today?"""
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Include system message and conversation history
        messages = [
            {"role": "system", "content": system_message}
        ] + self.history
        
        data = {
            "model": "mistral-small",  # Using a more capable model
            "messages": messages
        }
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            
            response_data = response.json()
            full_message = response_data['choices'][0]['message']['content']
            
            # Parse the pipe-separated format: emotion|response
            if "|" in full_message:
                parts = full_message.split("|", 1)
                emotion = parts[0].strip().lower()
                
                # Ensure valid emotion
                valid_emotions = self.available_moods
                if emotion not in valid_emotions:
                    emotion = self.mood  # Default to current mood if parsing fails
                else:
                    # Update the chatbot's current mood to match the response emotion
                    self.mood = emotion
                
                # Get the response content
                if len(parts) > 1:
                    content = parts[1].strip()
                else:
                    content = "I'm not sure what to say."
            else:
                # Fallback if format not followed
                content = full_message
                emotion = self.mood
            
            # Add assistant response to history (without the emotion part)
            self.history.append({"role": "assistant", "content": content})
            
            return content, emotion
        
        except requests.exceptions.RequestException as e:
            print(f"Error calling Mistral API: {e}")
            return "I'm sorry, I encountered an error while processing your request.", "sad"
    
    def _process_tts(self, text, emotion):
        """
        Process text through TTS and send audio data to the AudioGripperController.
        
        Args:
            text (str): Text to convert to speech
            emotion (str): Emotion of the response
        """
        try:
            # Generate audio using Kokoro
            generator = self.tts_pipeline(text, voice=self.voice)
            
            for _, _, audio in generator:
                # Convert PyTorch tensor to numpy array if needed
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().numpy()
                
                # Make sure audio is properly normalized
                audio_normalized = audio.astype(np.float32)
                if np.max(np.abs(audio_normalized)) > 0:  # Avoid division by zero
                    audio_normalized = audio_normalized / np.max(np.abs(audio_normalized))
                
                # Play audio and send it to the AudioGripperController
                sd.play(audio_normalized, samplerate=24000)
                self._send_audio_data(audio_normalized, emotion)
                
                # Wait for audio to finish playing
                sd.wait()
        
        except Exception as e:
            print(f"Error in TTS processing: {e}")
    
    def chat(self):
        """Start the chat interface in the terminal."""
        print("🤖 Robot Chatbot initialized! Type 'exit' to quit.")
        print(f"🤖 Current mood: {Fore.CYAN}{self.mood}{Style.RESET_ALL}")
        print(f"🤖 Use {Fore.CYAN}/mood [mood_name]{Style.RESET_ALL} to change my mood")
        print(f"🤖 Available moods: {Fore.CYAN}{', '.join(self.available_moods)}{Style.RESET_ALL}")
        
        if not self.sock:
            print(f"{Fore.RED}Warning: Unable to start socket server. Robot connection will not work.{Style.RESET_ALL}")
        
        while self.running:
            # Get user input
            user_input = input("\n🧑 You: ")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                self.running = False
                print("🤖 Robot: Goodbye!")
                break
                
            # Check if this is a command to change the mood
            if self._change_mood(user_input):
                continue
            
            # Get response from Mistral API
            print("\n🤖 Robot is thinking...")
            response, emotion = self._call_mistral_api(user_input)
            
            # Display the response with emotion
            print(f"🤖 Robot ({Fore.CYAN}{emotion}{Style.RESET_ALL}): {response}")
            
            # Process response through TTS
            self._process_tts(response, emotion)
    
    def close(self):
        """Close the chatbot and clean up resources."""
        self.running = False
        if self.sock:
            self.sock.close()


def main():
    """Main function to parse arguments and run the robot chatbot."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Robot Chatbot with TTS")
    parser.add_argument("--mistral-api-key", help="Mistral API key (can also use MISTRAL_API_KEY env variable)")
    parser.add_argument("--voice", default="am_fenrir", help="Voice for TTS")
    parser.add_argument("--lang-code", default="a", help="Language code")
    
    # Parse args
    args = parser.parse_args()
    
    # Get Mistral API key from arguments or environment variable
    mistral_api_key = args.mistral_api_key or os.environ.get("MISTRAL_API_KEY")
    
    if not mistral_api_key:
        print(f"{Fore.RED}Error: Mistral API key not provided. Please set the MISTRAL_API_KEY environment variable or use --mistral-api-key.{Style.RESET_ALL}")
        return
    
    # Initialize and start the chatbot
    chatbot = RobotChatbot(
        api_key=mistral_api_key,
        voice=args.voice,
        lang_code=args.lang_code
    )
    
    try:
        chatbot.chat()
    except KeyboardInterrupt:
        print("\nExiting chatbot...")
    finally:
        chatbot.close()


if __name__ == "__main__":
    main()