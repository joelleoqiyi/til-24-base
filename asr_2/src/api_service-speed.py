from fastapi import FastAPI, Request
import base64
from ASRManagerDoubleSpeed import ASRManager
from io import BytesIO
import librosa

app = FastAPI()

asr_manager = ASRManager()


@app.get("/health")
def health():
    return {"message": "health ok"}


@app.post("/stt")
async def stt(request: Request):
    """
    Performs ASR given the filepath of an audio file
    Returns transcription of the audio
    """

    # get base64 encoded string of audio, convert back into bytes
    input_json = await request.json()

    loaded_audio = []
    for instance in input_json["instances"]:
        # each is a dict with one key "b64" and the value as a b64 encoded string
        audio_bytes = base64.b64decode(instance["b64"])
        speech_array, sampling_rate = librosa.load(BytesIO(audio_bytes), sr=16000)
        loaded_audio.append(speech_array)

    predictions = asr_manager.transcribe(loaded_audio)


    return {"predictions": predictions}