from groq import Groq
import os 
from dotenv import load_dotenv
import base64
from deepgram import (
    DeepgramClient,
    SpeakOptions,
)
import requests
# import streamlit as st
load_dotenv()

client2 = Groq(
    api_key= os.getenv("LLMA_API_KEY")
)
DeepGram_API_KEY = os.getenv("DEEPGRAM_API_KEY")
deepgram = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))

def speech_to_text(audio_data):
    with open(audio_data, "rb") as audio_file:
        transcript = client2.audio.transcriptions.create(
            file=(audio_data, audio_file.read()),
            model="whisper-large-v3-turbo",
            prompt="Keep the filler words like \n\"Umm, let me think like, hmm... Okay, here's what I'm, like, thinking.\"\n\nInclude punctuation: \"Hello, welcome to my lecture.\"",
            language="en"
            )
        return transcript.text
    
def text_to_speech(text_response):
    # STEP 2: Configure the options (such as model choice, audio configuration, etc.)
    options = SpeakOptions(
        model="aura-asteria-en",
        encoding="linear16",
        container="wav"
    )

    SPEAK_OPTIONS = {"text": text_response}
    filename = "output.wav"
    response = deepgram.speak.v("1").save(filename, SPEAK_OPTIONS, options)
    return filename

# def autoplay_audio(file_path: str):
#     with open(file_path, "rb") as f:
#         data = f.read()
#     b64 = base64.b64encode(data).decode("utf-8")
#     md = f"""
#     <audio autoplay>
#     <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
#     </audio>
#     """
#     st.markdown(md, unsafe_allow_html=True)

def text_to_speech_streaming(text_response):
    DEEPGRAM_URL = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"

    payload = {
        "text": text_response

    }
    headers = {
        "Authorization": f"Token {DeepGram_API_KEY}",
        "Content-Type": "application/json"
    }

    audio_file_path = "output.wav"  # Path to save the audio file

    with open(audio_file_path, 'wb') as file_stream:
        response = requests.post(DEEPGRAM_URL, headers=headers, json=payload, stream=True)
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file_stream.write(chunk)  # Write each chunk of audio data to the file

    return audio_file_path


def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)