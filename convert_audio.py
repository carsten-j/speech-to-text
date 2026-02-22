from pydub import AudioSegment

audio = AudioSegment.from_file("sample.mp3")
audio.export("sample.wav", format="wav")
