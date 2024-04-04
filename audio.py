from faster_whisper import WhisperModel
from fastapi import FastAPI,UploadFile,File
import uvicorn
import librosa
app = FastAPI()
model = WhisperModel("large-v2")

@app.post("/Transcription")
async def root(file:UploadFile=File(...)):
    seg = " "
    audio_content,sr = librosa.load(file.file)
    segments, info = model.transcribe(audio_content)
    for segment in segments:
        seg += "%s " % segment.text
    return seg

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)