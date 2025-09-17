import os
import wave

from dotenv import load_dotenv
import sounddevice as sd
from pynput.keyboard import Controller as KeyboardController, Key, Listener

from wkey.whisper import apply_whisper
from wkey.utils import process_transcript, convert_chinese
from wkey.llm_correction import llm_corrector

load_dotenv()
key_label = os.environ.get("WKEY", "alt_l")
RECORD_KEY = Key[key_label]
CHINESE_CONVERSION = os.environ.get("CHINESE_CONVERSION")

# This flag determines when to record
recording = False

# This is where we'll store the audio (as bytes)
audio_data = []

# This is the sample rate for the audio
sample_rate = 16000

# Keyboard controller
keyboard_controller = KeyboardController()


def _record_and_transcribe():
    """Saves recorded audio and returns the transcript."""
    global audio_data
    
    if not audio_data:
        print("No audio data recorded, probably because the key was pressed for too short a time.")
        return None

    # Join the bytes chunks
    all_audio_bytes = b''.join(audio_data)
    
    # Write to a WAV file using the standard 'wave' module
    with wave.open('recording.wav', 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for 'int16'
        wf.setframerate(sample_rate)
        wf.writeframes(all_audio_bytes)

    try:
        transcript = apply_whisper('recording.wav', 'transcribe')
        return transcript
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def _process_and_type_transcript(transcript: str):
    """Applies corrections and conversions, then types the final transcript."""
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


def on_press(key):
    global recording
    global audio_data
    
    if key == RECORD_KEY:
        recording = True
        audio_data = []
        print("Listening...")

def on_release(key):
    global recording
    
    if key != RECORD_KEY:
        return

    recording = False
    print("Transcribing...")
    
    transcript = _record_and_transcribe()
    
    if transcript:
        _process_and_type_transcript(transcript)


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status)
    if recording:
        audio_data.append(indata) # indata is already a bytes object


def main():
    print(f"wkey is active. Hold down {key_label} to start dictating.")
    with Listener(on_press=on_press, on_release=on_release) as listener:
        # Use RawInputStream to get bytes directly, avoiding numpy conversion
        with sd.RawInputStream(callback=callback, channels=1, samplerate=sample_rate, dtype='int16'):
            listener.join()

if __name__ == "__main__":
    main()