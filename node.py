from PIL import Image
from transformers import pipeline
import torchvision.transforms as T
import torch
import numpy


class UnderageFilterNode:
    def __init__(self):
        self.classifier = pipeline("image-classification", model="nateraw/vit-age-classifier")
        self.to_pil = T.ToPILImage()
        self.underage_labels = {"0-2", "3-9", "10-19"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "score": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "age_confidence_threshold"
                }),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "check_underage"
    CATEGORY = "Classifier/Safety"

    def check_underage(self, image, score):
        img_tensor = image[0]
        pil_img = self.to_pil(img_tensor.permute(2, 0, 1))
        results = self.classifier(pil_img)

        top = max(results, key=lambda r: r["score"])
        is_underage = top["label"] in self.underage_labels and top["score"] >= score

        return (is_underage,)


NODE_CLASS_MAPPINGS = {
    "UnderageFilterNode": UnderageFilterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnderageFilterNode": "Underage Filter"
}
