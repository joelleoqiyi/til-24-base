from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import DatasetDict
import torchaudio
from torchaudio import transforms
from io import BytesIO


class ASRManager:
    def __init__(self):
        # initialize the model here
        # load model and processor
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperForConditionalGeneration.from_pretrained("./checkpoint-1000")
        
    def transcribe(self, audio_bytes: bytes) -> str:
        # perform ASR transcription
        speech_array, sampling_rate = torchaudio.load(BytesIO(audio_bytes))

        # resample to 16000 hz (required by model)
        if sampling_rate != 16000:
            transform = transforms.Resample(sampling_rate, 16000)
            speech_array = transform(speech_array)


        sample_audio = DatasetDict({
            'array': speech_array.squeeze(0),
            'sampling_rate': 16000
        })

        input_features = self.processor(sample_audio["array"], sampling_rate=sample_audio["sampling_rate"], return_tensors="pt").input_features 

        # generate predicted token ids
        predicted_ids = self.model.generate(input_features)
        # decode predicted token ids to text
        prediction = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        return prediction
