from typing import List
import io

import torch
from PIL import Image
from transformers import AutoProcessor, Owlv2ForObjectDetection, pipeline, AutoTokenizer, Owlv2ImageProcessor
import numpy as np

class VLMManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("./tokenizer_vlm")
        self.image_processor = Owlv2ImageProcessor()
        model_path = 'locallysavedmodel.pth'
        self.model = torch.load(model_path)
        self.pipe = pipeline(model=self.model, image_processor=self.image_processor, tokenizer=self.tokenizer, batch_size=16, device=self.device, task="zero-shot-object-detection", torch_dtype=torch.float16)
      

    def resize(self, img, base_width=380):
        wpercent = (base_width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((base_width, hsize), Image.Resampling.LANCZOS)
        return img

    def resize_box(self, box, original_size, resized_size):
        x_scale = original_size[0] / resized_size[0]
        y_scale = original_size[1] / resized_size[1]
        resized_box = [
            box['xmin'] * x_scale,
            box['ymin'] * y_scale,
            box['xmax'] * x_scale - box['xmin'] * x_scale,
            box['ymax'] * y_scale - box['ymin'] * y_scale
        ]
        return [int(round(coord, 0)) for coord in resized_box]

    def identify(self, image: bytes, caption: str) -> List[int]:
        image_stream = io.BytesIO(image)
        image = Image.open(image_stream).convert('RGB')
        ori_size = image.size
        # resized_image = self.resize(image)
        resized_image = image
        
        text = [caption.lower()]
        print(text)
        
        results = self.pipe(
            resized_image,
            candidate_labels=text,
        )
        
        x, y = ori_size
        final, best_score = [int(0.25*x), int(0.25*y), int(0.5*x), int(0.5*y)], 0
            
        for result in results:
            if result['score'] > best_score: 
                resized_box = self.resize_box(result['box'], ori_size, resized_image.size)
                final = resized_box
                best_score = result['score']

        return final