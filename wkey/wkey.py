import os

from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
from pynput.keyboard import Controller as KeyboardController, Key, Listener
from scipy.io import wavfile

from wkey.whisper import apply_whisper
from wkey.utils import process_transcript, convert_chinese
from wkey.llm_correction import llm_corrector

load_dotenv()
key_label = os.environ.get("WKEY", "alt_l")
RECORD_KEY = Key[key_label]
CHINESE_CONVERSION = os.environ.get("CHINESE_CONVERSION")

# This flag determines when to record
recording = False

# This is where we'll store the audio
audio_data = []

# This is the sample rate for the audio
sample_rate = 16000

# Keyboard controller
keyboard_controller = KeyboardController()


def on_press(key):
    global recording
    global audio_data
    
    if key == RECORD_KEY:
        recording = True
        audio_data = []
        print("Listening...")

def on_release(key):
    global recording
    global audio_data
    
    if key == RECORD_KEY:
        recording = False
        print("Transcribing...")
        
        try:
            audio_data_np = np.concatenate(audio_data, axis=0)
        except ValueError as e:
            print(e)
            return
        
        audio_data_int16 = (audio_data_np * np.iinfo(np.int16).max).astype(np.int16)
        wavfile.write('recording.wav', sample_rate, audio_data_int16)

        transcript = None
        try:
            transcript = apply_whisper('recording.wav', 'transcribe')
        except Exception as e:
            print(e)
        
        if transcript:
            is_llm_correct_enabled = os.getenv("LLM_CORRECT", "false").lower() in ("true", "1", "yes")
            if is_llm_correct_enabled:
                original_transcript = transcript
                transcript = llm_corrector(transcript)
                if original_transcript != transcript:
                    print(f"Before Corrected: {original_transcript}")

            if CHINESE_CONVERSION:
                transcript = convert_chinese(transcript, CHINESE_CONVERSION)
            processed_transcript = process_transcript(transcript)
            print(processed_transcript)
            keyboard_controller.type(processed_transcript)


def callback(indata, frames, time, status):
    if status:
        print(status)
    if recording:
        audio_data.append(indata.copy())  # make sure to copy the indata

def main():
    print(f"wkey is active. Hold down {key_label} to start dictating.")
    with Listener(on_press=on_press, on_release=on_release) as listener:
        # This is the stream callback
        with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate):
            # Just keep the script running
            listener.join()

if __name__ == "__main__":
    main()
