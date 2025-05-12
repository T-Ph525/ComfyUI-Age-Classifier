from PIL import Image
from transformers import pipeline
import torchvision.transforms as T
import torch
import numpy
import ast


def tensor2pil(image):
    return Image.fromarray(
        numpy.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(numpy.uint8)
    )


class UnderageBlockNode:
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

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Moderation/Blocker"

    def run(self, image, score):
        img_tensor = image[0]
        pil_img = self.to_pil(img_tensor.permute(2, 0, 1))
        results = self.classifier(pil_img)

        top = max(results, key=lambda r: r["score"])
        if top["label"] in self.underage_labels and top["score"] >= score:
            raise PermissionError(403, f"Underage content detected (label: {top['label']}, score: {top['score']:.3f})")

        return (image,)


NODE_CLASS_MAPPINGS = {
    "UnderageBlockNode": UnderageBlockNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnderageBlockNode": "Underage Block"
}
