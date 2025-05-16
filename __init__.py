from .node import UnderageFilterGateNode, UnderageFilterNode

NODE_CLASS_MAPPINGS = {
    "Underage Filter Node": UnderageFilterNode,
    "Underage Filter Gate": UnderageFilterGateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Underage Filter Node": "Underage Filter (Score Only)",
    "Underage Filter Gate": "Underage Filter Gate",
}

NODE_INPUT_TYPES = {
    "Underage Filter Node": {
        "image": "An image tensor (1xHxWx3) to analyze.",
    },
    "Underage Filter Gate": {
        "image": "The image to analyze and forward if safe.",
        "threshold": "Confidence score threshold. If exceeded, blocks content.",
    },
}

NODE_OUTPUT_TYPES = {
    "Underage Filter Node": {
        "score": "Confidence score (0-1) for underage detection.",
        "boolean": "True if flagged as underage.",
    },
    "Underage Filter Gate": {
        "image": "The input image if passed, otherwise blocks execution.",
    },
}

NODE_CATEGORIES = {
    "Underage Filter Node": "comfyui-underage-filter",
    "Underage Filter Gate": "comfyui-underage-filter",
}

