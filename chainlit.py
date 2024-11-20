import asyncio
from dotenv import load_dotenv
import os
import json
# from utils import text_to_speech
from utils import text_to_speech_streaming
from LLMA_Assistant import MasterDebtCollectorAssistantLLAMA
# from playsound import playsound
from pydub import AudioSegment
# from pydub.playback import play
from audio2 import play_audio
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)
import time

load_dotenv()

DeepGram_API_KEY = os.getenv("DEEPGRAM_API_KEY")
assistant = MasterDebtCollectorAssistantLLAMA()

class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

async def get_transcript():
    transcript_collector = TranscriptCollector()
    transcription_complete = asyncio.Event()
    microphone = None

    try:
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient(DeepGram_API_KEY, config)

        dg_connection = deepgram.listen.asynclive.v("1")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) == 0:
                return
            
            if result.is_final:
                transcript_collector.add_part(sentence)
                if result.speech_final:
                    print(".....speech final detected.....")
                    transcription_complete.set()

        async def on_error(self, error, **kwargs):
            print(f"\n\n{error}\n\n")

        async def on_utterance_end(self, **kwargs):
            # No need to reset the transcript here
            #pass
            if len(transcript_collector.transcript_parts) > 0:
                print(".....utterance end detected.....")
                transcription_complete.set()

        # Register event handlers
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            interim_results=True,
            utterance_end_ms="1000",
            vad_events=True,
            endpointing=300
        )

        # Start the connection
        if not await dg_connection.start(options):
            print('Failed to connect to Deepgram')
            return

        # Open a microphone stream on the default input device
        microphone = Microphone(dg_connection.send)

        # Start microphone
        microphone.start()

        while True:
            # Wait for transcription to complete
            await transcription_complete.wait()
            transcription_complete.clear()
            full_sentence = transcript_collector.get_full_transcript()
            transcript_collector.reset()

            if not full_sentence.strip():
                continue

            print(".....long pause detected.....")
            print(f"Consumer: {full_sentence}")

            # Mute the microphone during TTS playback
            microphone.mute()
            # note start time
            start_time = time.time()
            start_time1 = time.time()

            # Send the user's input to the LLM assistant
            loop = asyncio.get_event_loop()
            assistant_response = await loop.run_in_executor(None, assistant.get_assistant_response, full_sentence)
            assistant_json_response, transition, context = assistant_response

            # note end time
            end_time1 = time.time()
            # Print the time taken for assistant response
            print(f"Assistant Latency: {end_time1 - start_time1} seconds")

            assistant_dict_response = json.loads(assistant_json_response)
            collector_response = assistant_dict_response["Debt Collector"]

            start_time2 = time.time()
            # Convert the collector's response to speech and play it
            audio_file = await loop.run_in_executor(None, text_to_speech_streaming, collector_response)
            end_time2 = time.time()
            # Print the time taken to generate audio 
            print(f"Audio Generation Latency: {end_time2 - start_time2} seconds")
            # audio = AudioSegment.from_file(audio_file, format="mp3")
            await loop.run_in_executor(None, play_audio, audio_file)

            # Play the audio file directly without loading in memory
            # await loop.run_in_executor(None, playsound, audio_file)
            
            # note end time
            end_time = time.time()
            # Unmute the microphone
            microphone.unmute()

            # Print the collector's response
            print(f"Debt Collector: {collector_response}")
            # Print the time taken to process the response
            print(f"Overall Latency: {end_time - start_time} seconds")

            # Remove the file after playing
            if os.path.exists(audio_file):
                os.remove(audio_file)

            if os.path.exists('output_converted.wav'):
                os.remove('output_converted.wav')

    except Exception as e:
        print(f"Could not open socket: {e}")
        return
    finally:
        if microphone:
            microphone.finish()
        await dg_connection.finish()
        print("Finished")

def start_initial_prompt():
    # Send initial prompt to assistant
    initial_prompt = (
        "Start conversation, Start the outbound call, You as debt collector has some "
        "consumer information on file : consumer name John Smith ,US consumer (Georgia) "
        "of balance 1200$ for medical bill, progressive emergency physicians bills , "
        "service date july 17,2019, SSN - 123-45-6789, DOB - 27-03-1974."
    )
    assistant_json_response, transition, context = assistant.get_assistant_response(initial_prompt)
    assistant_dict_response = json.loads(assistant_json_response)
    collector_response = assistant_dict_response["Debt Collector"]

    # Convert the collector's response to speech and play it
    audio_file = text_to_speech_streaming(collector_response)

    # audio = AudioSegment.from_file(audio_file, format="mp3")
    play_audio(audio_file)
    # playsound(audio_file)

    # Remove the file after playing
    if os.path.exists(audio_file):
        os.remove(audio_file)

    if os.path.exists('output_converted.wav'):
        os.remove('output_converted.wav')

    # Print the collector's response
    print(f"Debt Collector: {collector_response}")

def main():
    start_initial_prompt()
    asyncio.run(get_transcript())

if __name__ == "__main__":
    main()
