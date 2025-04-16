import streamlit as st
from openai import OpenAI
import os
from tempfile import gettempdir
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) # We ignore the warning for stream_to_file method

# Ensure you have your OpenAI API key in Streamlit secrets
# It will look for st.secrets['OPENAI_API_KEY']
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    st.error(f"Failed to initialize OpenAI client: {e}. Make sure your OPENAI_API_KEY is set in Streamlit secrets.", icon="🚨")
    client = None

def synthesize_speech(text: str, voice: str = "alloy") -> str | None:
    """
    Synthesizes speech from text using OpenAI's TTS API and saves it to a temporary file.

    Args:
        text (str): The text to synthesize.
        voice (str): The voice model to use (e.g., 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'). Defaults to "alloy".

    Returns:
        str | None: The file path to the generated MP3 audio file, or None if an error occurred.
    """
    if not client:
        st.error("OpenAI client not initialized. Cannot synthesize speech.")
        return None
    if not text:
        st.warning("No text provided for speech synthesis.")
        return None

    try:
        # Define the path for the temporary audio file
        # Using Pathlib for better path handling
        temp_dir = Path(gettempdir())
        output_path = temp_dir / "speech.mp3"

        # Make the API call to OpenAI TTS
        response = client.audio.speech.create(
            model="tts-1",  # You can also try "tts-1-hd" for higher definition (different pricing)
            voice=voice,
            input=text,
            response_format="mp3" # Ensure format is mp3
        )

        # Stream the audio response to the file
        # The 'response' object itself doesn't have a '.read()' method like boto3's stream.
        # Instead, use response.stream_to_file or response.write_to_file
        response.stream_to_file(str(output_path)) # Use str() as stream_to_file expects a string path

        return str(output_path)

    except Exception as error:
        st.error(f"Error synthesizing speech with OpenAI: {error}", icon="🚨")
        # Could not write to file or API error, return None
        print(f"OpenAI TTS Error: {error}")
        return None

# Example usage (optional, for testing the function directly)
# if __name__ == '__main__':
#     # Make sure to set OPENAI_API_KEY environment variable or use Streamlit secrets if running within Streamlit
#     st.secrets["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE" # Replace with your key for direct testing
#     test_text = "Hello, this is a test of OpenAI Text-to-Speech."
#     file_path = synthesize_speech(test_text)
#     if file_path:
#         print(f"Speech synthesized successfully and saved to: {file_path}")
#         # You would typically play this file using an appropriate library or command
#     else:
#         print("Speech synthesis failed.")