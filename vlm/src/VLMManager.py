from typing import List


class VLMManager:
    def __init__(self):
        # initialize the model here
        # jt: think these 2 are worth trying: https://huggingface.co/google/owlv2-base-patch16-ensemble
        # and https://huggingface.co/IDEA-Research/grounding-dino-tiny
        
        pass

    def identify(self, image: bytes, caption: str) -> List[int]:
        # perform object detection with a vision-language model
        return [0, 0, 0, 0]
