# Install required libraries (run in a notebook or shell environment, not in a script)
!pip install gradio openai gtts pydub numpy requests groq openai-whisper
!apt-get install -y ffmpeg

# Import necessary libraries
import os
import gradio as gr                     # For building a web UI
import whisper                          # OpenAI's Whisper model for audio transcription
from gtts import gTTS                   # Google Text-to-Speech
from groq import Groq, GroqError        # Groq API client for LLaMA models
from typing import Tuple, Union

# Load Whisper model (base version)
model = whisper.load_model("base")

# Initialize Groq API client with your API key
api_key = "groq_api_key"  # Replace with your actual API key
try:
    client = Groq(api_key=api_key)
except Exception as e:
    raise RuntimeError(f"Failed to initialize Groq client: {e}")

# Function that handles transcription and response generation
def transcribe_and_respond(audio: str) -> Tuple[str, Union[str, None]]:
    try:
        # Step 1: Transcribe audio file using Whisper
        transcription = model.transcribe(audio)
        user_input = transcription['text']  # Get the text from the transcription

        # Step 2: Generate response using Groq's LLaMA model
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": user_input}],
                model="llama3-8b-8192",
            )
            response_text = chat_completion.choices[0].message.content
        except GroqError as e:
            return f"Error in Groq API call: {e}", None

        # Step 3: Convert the response text to speech using gTTS
        tts = gTTS(response_text)
        audio_path = "response.mp3"
        tts.save(audio_path)

        return response_text, audio_path  # Return both the response text and audio

    except FileNotFoundError:
        return "Error: Audio file not found.", None
    except whisper.WhisperError as e:
        return f"Error in transcription: {e}", None
    except Exception as e:
        return f"An unexpected error occurred: {e}", None

# Set up a Gradio interface for interactive use
interface = gr.Interface(
    fn=transcribe_and_respond,
    inputs=gr.Audio(type="filepath"),
    outputs=[gr.Textbox(label="Response"), gr.Audio(label="Voice Response")],
    live=True
)

# Launch the interface
interface.launch()# Install required libraries (run in a notebook or shell environment, not in a script)
!pip install gradio openai gtts pydub numpy requests groq openai-whisper
!apt-get install -y ffmpeg

# Import necessary libraries
import os
import gradio as gr                     # For building a web UI
import whisper                          # OpenAI's Whisper model for audio transcription
from gtts import gTTS                   # Google Text-to-Speech
from groq import Groq, GroqError        # Groq API client for LLaMA models
from typing import Tuple, Union

# Load Whisper model (base version)
model = whisper.load_model("base")

# Initialize Groq API client with your API key
api_key = "groq_api_key"  # Replace with your actual API key
try:
    client = Groq(api_key=api_key)
except Exception as e:
    raise RuntimeError(f"Failed to initialize Groq client: {e}")

# Function that handles transcription and response generation
def transcribe_and_respond(audio: str) -> Tuple[str, Union[str, None]]:
    try:
        # Step 1: Transcribe audio file using Whisper
        transcription = model.transcribe(audio)
        user_input = transcription['text']  # Get the text from the transcription

        # Step 2: Generate response using Groq's LLaMA model
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": user_input}],
                model="llama3-8b-8192",
            )
            response_text = chat_completion.choices[0].message.content
        except GroqError as e:
            return f"Error in Groq API call: {e}", None

        # Step 3: Convert the response text to speech using gTTS
        tts = gTTS(response_text)
        audio_path = "response.mp3"
        tts.save(audio_path)

        return response_text, audio_path  # Return both the response text and audio

    except FileNotFoundError:
        return "Error: Audio file not found.", None
    except whisper.WhisperError as e:
        return f"Error in transcription: {e}", None
    except Exception as e:
        return f"An unexpected error occurred: {e}", None

# Set up a Gradio interface for interactive use
interface = gr.Interface(
    fn=transcribe_and_respond,
    inputs=gr.Audio(type="filepath"),
    outputs=[gr.Textbox(label="Response"), gr.Audio(label="Voice Response")],
    live=True
)

# Launch the interface
interface.launch()
