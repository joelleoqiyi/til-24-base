from typing import List
import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import io

import json

from PIL import Image
import numpy as np
from transformers import Owlv2ForObjectDetection
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
print('update5')

class VLMManager:
    def __init__(self):
        # initialize the model here
        # jt: https://huggingface.co/IDEA-Research/grounding-dino-tiny
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        '''
        self.model_id = "IDEA-Research/grounding-dino-tiny"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)
        '''
        self.processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model.to(self.device)

    def resize(self, img, base_width=380):
        wpercent = (base_width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((base_width, hsize), Image.Resampling.LANCZOS)
        return img
    
    def resize_box(self, box, original_size, resized_size):
        x_scale = original_size[0] / resized_size[0]
        y_scale = original_size[1] / resized_size[1]
        resized_box = [
            box[0] * x_scale,
            box[1] * y_scale,
            box[2] * x_scale,
            box[3] * y_scale
        ]
        return [round(coord, 0) for coord in resized_box]
        
    def identify(self, image: bytes, caption: str) -> List[int]:
        # perform object detection with a vision-language model, 
        image_stream = io.BytesIO(image)
        image = Image.open(image_stream).convert('RGB')
        ori_size = image.size
        unnormalized_image = self.resize(image)
        
        # text = "a yellow helicopter."
        text = [["a photo of" + caption.lower()]]
        
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        '''
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )

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
        '''
        box = results[0]['boxes'][0]
        box = [int(coord) for coord in box.tolist()]
        '''
        unnormalized_image = image

        target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to final bounding boxes and scores
        results = self.processor.post_process_object_detection(
            outputs=outputs, threshold=0.2, target_sizes=target_sizes
        )

        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = text[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        
        final, best = None, 0
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            resized_box = self.resize_box(box, ori_size, unnormalized_image.size)
            
            final = resized_box if score.item() > best else final
            best = max(best, score.item())
            
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {resized_box}")
        return final

    
from typing import List
import io

import torch
from PIL import Image
from transformers import AutoProcessor, Owlv2ForObjectDetection
import numpy as np

class VLMManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model.to(self.device)

    def resize(self, img, base_width=380):
        wpercent = (base_width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((base_width, hsize), Image.Resampling.LANCZOS)
        return img

    def resize_box(self, box, original_size, resized_size):
        x_scale = original_size[0] / resized_size[0]
        y_scale = original_size[1] / resized_size[1]
        resized_box = [
            box[0] * x_scale,
            box[1] * y_scale,
            box[2] * x_scale,
            box[3] * y_scale
        ]
        return [int(round(coord, 0)) for coord in resized_box]

    def identify(self, image: bytes, caption: str) -> List[int]:
        image_stream = io.BytesIO(image)
        image = Image.open(image_stream).convert('RGB')
        ori_size = image.size
        resized_image = self.resize(image)
        
        text = [["a photo of " + caption.lower()]]
        
        inputs = self.processor(images=resized_image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.Tensor([resized_image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs=outputs, threshold=0.2, target_sizes=target_sizes
        )

        i = 0
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        
        final, best_score = None, 0
        for box, score, label in zip(boxes, scores, labels):
            if score.item() > best_score:
                resized_box = self.resize_box(box.tolist(), ori_size, resized_image.size)
                final = resized_box
                best_score = score.item()

        return final
    
print('start')
vlm = VLMManager()
# Define the path to the image
image_path = 'example.jpg'
# Open the file in binary mode ('rb')
with open(image_path, 'rb') as file:
    image_bytes = file.read()  # Read the entire file as a bytes object
    
result = vlm.identify(image_bytes, 'yellow helicopter')
print(result)