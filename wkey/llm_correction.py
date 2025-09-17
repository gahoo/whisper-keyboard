import os
import requests
from collections import deque

def create_llm_corrector():
    """
    Creates an LLM text corrector that maintains a history of the last 10 transcripts.
    """
    history = deque(maxlen=10)
    
    api_url = os.getenv("LLM_CORRECT_API_URL")
    api_key = os.getenv("LLM_CORRECT_API_KEY")
    model = os.getenv("LLM_CORRECT_MODEL", "gpt-4")

    def corrector(transcript: str) -> str:
        """
        Sends the conversation history and the latest transcript to an LLM for correction.
        """
        if not transcript.strip():
            return transcript

        # The context is the history of previous corrected transcripts.
        context = "".join(history)
        text_to_correct = transcript
        
        prompt = f"""You are a text correction expert. Your task is to correct the following text by adding punctuation, fixing typos, and improving grammar. The text is a transcript of spoken words. Do not add any explanations, preambles, or apologies. Only return the corrected version of the text.

Conversation History (for context):
{context}

Text to Correct:
{text_to_correct}

Corrected Text:
"""

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 4096,
            "stop": ["\n"]
        }

        try:
            print("Correcting...")
            response = requests.post(api_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            corrected_transcript = response.json()["choices"][0]["message"]["content"].strip()
            
            # Add the corrected transcript to the history for the next turn's context.
            history.append(corrected_transcript)
            
            return corrected_transcript

        except requests.RequestException as e:
            print(f"Error calling LLM API: {e}")
            return transcript # Return original transcript on error
        except (KeyError, IndexError) as e:
            print(f"Error parsing LLM response: {e}")
            return transcript # Return original transcript on error

    return corrector

llm_corrector = create_llm_corrector()
