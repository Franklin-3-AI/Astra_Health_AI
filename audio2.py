from pydub import AudioSegment
import winsound

def play_audio(audio_file):
    output_file = 'output_converted.wav'
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_frame_rate(44100)      # Set sample rate to 44.1 kHz
    audio = audio.set_channels(2)            # Set to stereo
    audio = audio.set_sample_width(2)
    audio.export(output_file, format='wav')
    winsound.PlaySound(output_file, winsound.SND_FILENAME)