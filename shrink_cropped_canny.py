import torch
import numpy as np
from PIL import Image

# SHRINK CANNY IMAGE NODE

class ShrinkCannyImage:
    def __init__(self):
        pass 

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "canny_image": ("IMAGE",),  # Input cropped Canny image
                "shrink_factor": ("FLOAT", {  # Editable shrink factor
                    "default": 0.9,  # Default shrink factor is 90% of original size
                    "min": 0.1,  # Minimum shrink factor is 10%
                    "max": 1.0,  # Maximum shrink factor is 100%
                    "step": 0.01,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "shrink_image"

    CATEGORY = "REMADE/Image"

    def shrink_image(self, canny_image, shrink_factor):
        # Convert tensor to PIL image for processing
        canny_image_pil = tensor2pil(canny_image).convert("L")  # Convert to grayscale

        # Get the original size of the Canny image
        original_width, original_height = canny_image_pil.size

        # Calculate the new dimensions based on the shrink factor
        new_width = int(original_width * shrink_factor)
        new_height = int(original_height * shrink_factor)

        # Resize the image using the calculated dimensions
        shrunk_canny_image = canny_image_pil.resize((new_width, new_height), Image.LANCZOS)

        # Return the shrunk image without pasting it on a canvas
        result_tensor = pil2tensor(shrunk_canny_image)

        return (result_tensor, )

# Helper functions to convert between PIL and tensor

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "Shrink Canny Image": ShrinkCannyImage
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Shrink Canny Image": "Shrink Canny Image"
}
