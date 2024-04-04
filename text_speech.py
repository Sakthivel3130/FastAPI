
import uvicorn
from TTS.api import TTS
from fastapi import FastAPI
app = FastAPI()

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

def text_to_speech(text):
    output_file_path = "output.wav"
    language = "en"
    speaker_wav = "sample2.wav"
    tts.tts_to_file(text=text, file_path=output_file_path, speaker_wav=speaker_wav, language=language)
    return output_file_path

@app.post("/generate_speech")
async def generate_speech(text: str):
    output_file_path = text_to_speech(text)
    return {"speech_file_path": output_file_path}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
