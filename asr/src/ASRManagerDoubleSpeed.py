from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from io import BytesIO
import librosa  
import torch
import numpy.typing as npt

class ASRManager:
    def __init__(self):
        # initialize the model here
        # load model and processor
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "./checkpoint-1000"

        model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = WhisperProcessor.from_pretrained("./processor_asr", language="English", task="transcribe")

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=False,
            torch_dtype=torch_dtype,
            device=device,
        )

    def transcribe(self, audio_bytes: [npt.NDArray]) -> [str]:
        # perform ASR transcription
        result = self.pipe(audio_bytes)
        return [i['text'] for i in result]
