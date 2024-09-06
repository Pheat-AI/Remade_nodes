import torch
import numpy as np
from PIL import Image

# IMAGE REMOVE BLACK NODE

class REMADE_Remove_Black_To_Transparent:
    def __init__(self):
        pass 

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",)
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_black_pixels"

    CATEGORY = "REMADE/Image"

    def remove_black_pixels(self, image):
        # Convert the input tensor to a PIL image
        image = tensor2pil(image).convert("RGBA")

        # Convert image data to numpy array for pixel manipulation
        data = np.array(image)

        # Create a mask where all black pixels (0, 0, 0) will be marked for transparency
        # The last channel (A) will be set to 0 for transparency
        black_pixels = (data[:, :, 0] == 0) & (data[:, :, 1] == 0) & (data[:, :, 2] == 0)
        data[black_pixels, 3] = 0  # Set alpha channel to 0 for pure black pixels

        # Convert the numpy array back to a PIL image
        result_image = Image.fromarray(data)

        # Convert back to tensor
        result_tensor = pil2tensor(result_image)

        return (result_tensor, )

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Remove Black To Transparent": REMADE_Remove_Black_To_Transparent
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Remove Black To Transparent": "Remove Black Pixels to Transparent"
}
