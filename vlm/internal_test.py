from typing import List

import requests
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, Owlv2ForObjectDetection
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

import io
from torchvision import transforms

class VLMManager:
    def __init__(self):
        # initialize the model here
        # jt: think these 2 are worth trying: https://huggingface.co/google/owlv2-base-patch16-ensemble
        # and https://huggingface.co/IDEA-Research/grounding-dino-tiny
        self.processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        
        pass
    def byte_to_tensor(self, image: bytes):
        image_stream = io.BytesIO(image)
        image = Image.open(image_stream).convert('RGB')
        image.save('new_image.jpg')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Common normalization
        ])
        image_tensor = transform(image)
        print(image_tensor.shape)
        return image_tensor

    def identify(self, image: bytes, caption: str) -> List[int]:
        print('Updated3')
        # perform object detection with a vision-language model
        image = self.byte_to_tensor(image)
        inputs = self.processor(text=caption, images=image, return_tensors="pt")
        
        # forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Note: boxes need to be visualized on the padded, unnormalized image
        # hence we'll set the target image sizes (height, width) based on that

        def get_preprocessed_image(pixel_values):
            pixel_values = pixel_values.squeeze().numpy()
            unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
            unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
            unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
            unnormalized_image = Image.fromarray(unnormalized_image)
            return unnormalized_image

        unnormalized_image = get_preprocessed_image(inputs.pixel_values)

        target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to final bounding boxes and scores
        results = self.processor.post_process_object_detection(
            outputs=outputs, threshold=0.01, target_sizes=target_sizes
        )
        
        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        box, score, label = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        print(box)
        box = [round(coord, 2) for coord in box.tolist()]
        print(box)
        print(f"Detected {caption} with confidence {round(score.item(), 3)} at location {box}")
        return box
    
# Define the path to the image
image_path = 'example.jpg'
# Open the file in binary mode ('rb')
with open(image_path, 'rb') as file:
    image_bytes = file.read()  # Read the entire file as a bytes object
# Now, image_bytes contains the image as a byte array
print(type(image_bytes))  # Should print <class 'bytes'>

vlm = VLMManager()
output = vlm.identify(image_bytes, "yellow helicopter")
print(output)