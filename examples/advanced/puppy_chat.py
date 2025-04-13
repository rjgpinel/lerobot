import time
import requests
import json
import os
from collections import deque
import threading
import argparse
from kokoro import KPipeline
import sounddevice as sd
import numpy as np
import torch
from colorama import Fore, Style, init


class PuppyChatbot:
    def __init__(self, api_key, lip_sync_callback, voice="af_heart", lang_code="a"):
        """
        Initialize the Puppy chatbot with Mistral API and Kokoro TTS.
        
        Args:
            api_key (str): Mistral API key
            lip_sync_callback (function): Callback function for lip sync, called with audio chunks
            voice (str, optional): Voice to use for TTS. Defaults to "af_heart".
            lang_code (str, optional): Language code. Defaults to "a".
        """
        self.api_key = api_key
        self.lip_sync_callback = lip_sync_callback
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
        
        # Initialize the audio buffer
        self.audio_buffer = deque()
        
        # Initialize Kokoro TTS pipeline
        self.tts_pipeline = KPipeline(lang_code=self.lang_code)
        
        # Message history for context
        self.history = []
        
        # Start the lip sync thread
        self.running = True
        self.lip_sync_thread = threading.Thread(target=self._lip_sync_worker)
        self.lip_sync_thread.daemon = True
        self.lip_sync_thread.start()

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
            print(f"{Fore.CYAN}🐶 Mood changed from {old_mood} to {self.mood}{Style.RESET_ALL}")
            return True
        else:
            # List available moods
            moods_list = ", ".join(self.available_moods)
            print(f"{Fore.YELLOW}Available moods: {moods_list}{Style.RESET_ALL}")
            return True  # Still handled the command, even if mood didn't change

    def _lip_sync_worker(self):
        """Worker thread that calls the lip sync callback every 1ms with new audio data."""
        while self.running:
            if self.audio_buffer:
                # Get audio chunk from buffer
                audio_chunk = self.audio_buffer.popleft()
                # Call the lip sync callback with the audio chunk
                self.lip_sync_callback(audio_chunk)
            # Sleep for 1ms
            time.sleep(0.001)
    
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
        system_message = f"""You are a puppy chatbot at an AI hackathon. Your starting mood is {self.mood.upper()}. ONLY respond with the exact format below - nothing else.

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
                    content = "Woof! I'm not sure what to say."
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
    
    def _process_tts(self, text):
        """
        Process text through TTS and add chunks to the audio buffer.
        
        Args:
            text (str): Text to convert to speech
        """
        try:
            # Generate audio using Kokoro
            generator = self.tts_pipeline(text, voice=self.voice)
            
            for _, _, audio in generator:
                # Convert PyTorch tensor to numpy array if needed
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().numpy()
                
                # Play audio directly using sounddevice
                # Make sure audio is properly normalized
                audio_normalized = audio.astype(np.float32)
                if np.max(np.abs(audio_normalized)) > 0:  # Avoid division by zero
                    audio_normalized = audio_normalized / np.max(np.abs(audio_normalized))
                
                # Play audio in a non-blocking way
                sd.play(audio_normalized, samplerate=24000)
                
                # Split audio into smaller chunks for smoother lip sync
                chunk_size = 240  # 10ms at 24kHz
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i+chunk_size]
                    if len(chunk) > 0:
                        self.audio_buffer.append(chunk)
        
        except Exception as e:
            print(f"Error in TTS processing: {e}")
    
    def chat(self):
        """Start the chat interface in the terminal."""
        print("🐶 Puppy Chatbot initialized! Type 'exit' to quit.")
        print(f"🐶 Current mood: {Fore.CYAN}{self.mood}{Style.RESET_ALL}")
        print(f"🐶 Use {Fore.CYAN}/mood [mood_name]{Style.RESET_ALL} to change my mood")
        print(f"🐶 Available moods: {Fore.CYAN}{', '.join(self.available_moods)}{Style.RESET_ALL}")
        
        while True:
            # Get user input
            user_input = input("\n🧑 You: ")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                self.running = False
                print("🐶 Puppy: Goodbye!")
                break
                
            # Check if this is a command to change the mood
            if self._change_mood(user_input):
                continue
            
            # Get response from Mistral API
            print("\n🐶 Puppy is thinking...")
            response, emotion = self._call_mistral_api(user_input)
            
            # Display the response with emotion
            print(f"🐶 Puppy ({Fore.CYAN}{self.mood}{Style.RESET_ALL}): {response} {Fore.RED}<{emotion}>{Style.RESET_ALL}")
            
            # Process response through TTS in a separate thread to not block the main thread
            tts_thread = threading.Thread(target=self._process_tts, args=(response,))
            tts_thread.daemon = True
            tts_thread.start()
            


def dummy_lip_sync_callback(audio_chunk):
    """
    Dummy callback function for lip sync. Replace this with your actual implementation.
    
    Args:
        audio_chunk: Audio chunk data
    """
    # Just a placeholder - this will be replaced by the user's actual lip sync function
    pass

def main():
    parser = argparse.ArgumentParser(description="Puppy Chatbot with TTS and Lip-Sync")
    parser.add_argument("--api-key", help="Mistral API key (can also use MISTRAL_API_KEY env variable)")
    parser.add_argument("--voice", default="af_heart", help="Voice for TTS")
    parser.add_argument("--lang-code", default="a", help="Language code")
    args = parser.parse_args()
    
    # Get API key from arguments or environment variable
    api_key = args.api_key or os.environ.get("MISTRAL_API_KEY")
    
    if not api_key:
        print(f"{Fore.RED}Error: Mistral API key not provided. Please set the MISTRAL_API_KEY environment variable or use --api-key.{Style.RESET_ALL}")
        return
    
    # Initialize and start the chatbot
    # Replace dummy_lip_sync_callback with your actual lip sync function
    chatbot = PuppyChatbot(
        api_key=api_key, 
        lip_sync_callback=dummy_lip_sync_callback,
        voice=args.voice,
        lang_code=args.lang_code
    )
    chatbot.chat()

if __name__ == "__main__":
    main()