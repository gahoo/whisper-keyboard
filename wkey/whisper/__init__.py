import os
from dotenv import load_dotenv

load_dotenv()

backend = os.environ.get("WHISPER_BACKEND", "groq")

if backend == "openai":
    from .openai import apply_whisper
elif backend == "groq":
    from .groq import apply_whisper
elif backend == "whisperx":
    from .whisperx import apply_whisper
elif backend == "insanely-whisper":
    from .insanely_whisper import apply_whisper
else:
    raise ImportError(f"Invalid whisper backend: {backend}")
