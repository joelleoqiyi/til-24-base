from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from io import BytesIO
import librosa  
import torch

class ASRManager:
    def __init__(self):
        # initialize the model here
        # load model and processor
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "./checkpoint-8000"

        model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")

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

    def transcribe(self, audio_bytes: bytes) -> str:
        # perform ASR transcription
        speech_array,_ = librosa.load(BytesIO(audio_bytes), sr=16000) # downsample to sr 16000
        result = self.pipe(speech_array)
        return result['text']