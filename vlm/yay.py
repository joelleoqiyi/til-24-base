from typing import List
import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import io

import json

class VLMManager:
    def __init__(self):
        # initialize the model here
        # jt: https://huggingface.co/IDEA-Research/grounding-dino-tiny
        self.model_id = "IDEA-Research/grounding-dino-tiny"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)

    def identify(self, image: bytes, caption: str) -> List[int]:
        # perform object detection with a vision-language model, 
        image_stream = io.BytesIO(image)
        image = Image.open(image_stream).convert('RGB')
        
        # text = "a yellow helicopter."
        text = caption.lower() + "."
        
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
        '''
        results = [{
            'scores': torch.tensor([0.4785, 0.4379, 0.4761]),
            'labels': ['a cat', 'a cat', 'a remote control'],
            'boxes': torch.tensor([
                [344.6982,  23.1083, 637.1821, 374.2747],
                [ 12.2693,  51.9104, 316.8566, 472.4341],
                [ 38.5852,  70.0090, 176.7768, 118.1755]
            ])
        }]
        '''
        box = results[0]['boxes'][0]
        box = [int(coord) for coord in box.tolist()]
        return box
    
vlm = VLMManager()
# Define the path to the image
image_path = 'example.jpg'
# Open the file in binary mode ('rb')
with open(image_path, 'rb') as file:
    image_bytes = file.read()  # Read the entire file as a bytes object
    
result = vlm.identify(image_bytes, 'yellow helicopter')
print(result)