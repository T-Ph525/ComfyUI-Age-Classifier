from transformers import pipeline
from PIL import Image
import torch
import os

class UnderageFilterGateNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "comfyui-underage-filter"

    def __init__(self):
        # Load once
        self.classifier = None
        self.model_path = "/comfy/models/age-classifier"
        self.ensure_model()

    def ensure_model(self):
        if self.classifier is None:
            self.classifier = pipeline(
                "image-classification",
                model=self.model_path,
                device=0 if torch.cuda.is_available() else -1,
            )

    def process(self, image, threshold):
        # Convert tensor to PIL
        image_pil = Image.fromarray((image[0].cpu().numpy() * 255).astype("uint8"))

        # Run classification
        result = self.classifier(image_pil)[0]
        score = result["score"]
        label = result["label"]

        if "under" in label.lower() and score >= threshold:
            raise PermissionError(f"403 Forbidden: Image flagged as underage (score={score:.2f})")

        return (image,)
