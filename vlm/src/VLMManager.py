from typing import List
import io

import torch
from PIL import Image
from transformers import AutoProcessor, Owlv2ForObjectDetection
import numpy as np

class VLMManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained("./processor_vlm")
        # replaced: self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        # ADDED
        model_path = 'locallysavedmodel.pth'
        self.model = torch.load(model_path)
        # END OF ADDED
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
            box[2] * x_scale - box[0] * x_scale,
            box[3] * y_scale - box[1] * y_scale
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
        
        
        x, y = ori_size
        final, best_score = [int(0.25*x), int(0.25*y), int(0.5*x), int(0.5*y)], 0
        for box, score, label in list(zip(boxes, scores, labels)):
            if score.item() > best_score:
                # print(caption, box, score, label)
                resized_box = self.resize_box(box.tolist(), ori_size, resized_image.size)
                final = resized_box
                best_score = score.item()
                # print(caption, resized_box)
        
    
        # arr = list(zip(boxes, scores))
        # print(arr)
        # sorted_array = sorted(arr, key=lambda x: x[2].item(), reverse=True)
        # box, score, label = sorted_array[0]
        # resized_box = self.resize_box(box.tolist(), ori_size, resized_image.size)
        # final = resized_box
        # best_score = score.item()

        return final