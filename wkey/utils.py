import opencc

def process_transcript(transcript: str):
    return transcript + " "

def convert_chinese(text: str, conversion: str) -> str:
    """Converts text between Simplified and Traditional Chinese."""
    try:
        converter = opencc.OpenCC(f'{conversion}')
        return converter.convert(text)
    except Exception as e:
        print(f"Error during Chinese conversion: {e}")
        return text
