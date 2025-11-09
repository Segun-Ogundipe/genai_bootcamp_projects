import os
import tempfile

from gtts import gTTS
import whisper

class VoiceProcessor:
    def __init__(self):
        self.whisper_model = whisper.load_model("base")

    def transcribe(self, audio_value):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_value.getvalue())
            result = self.whisper_model.transcribe(temp_audio.name)
            os.unlink(temp_audio.name)

        return result["text"]

    def text_to_speech(self, text):
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts.save(temp_audio.name)
            audio_file = open(temp_audio.name, "rb")
            audio_bytes = audio_file.read()
            os.unlink(temp_audio.name)
        return audio_bytes
        