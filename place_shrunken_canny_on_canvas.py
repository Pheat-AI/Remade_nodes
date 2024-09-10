import torch
import numpy as np
from PIL import Image
import comfy.model_management  # Import for handling device management

# PLACE CANNY ON CENTERED CANVAS NODE

class PlaceCannyOnCanvas:
    def __init__(self):
        pass 

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shrunken_canny_image": ("IMAGE",),  # The shrunken Canny image
                "center_x": ("INT",),  # x-coordinate of the center
                "center_y": ("INT",),  # y-coordinate of the center
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "place_on_canvas"

    CATEGORY = "REMADE/Image"

    def place_on_canvas(self, shrunken_canny_image, center_x, center_y):
        # Convert tensor to PIL image for processing, ensure it is RGB format
        canny_image_pil = tensor2pil(shrunken_canny_image).convert("RGB")  # Convert to RGB for 3 channels

        # Get the dimensions of the shrunken Canny image
        canny_width, canny_height = canny_image_pil.size

        # Create a new 1024x1024 blank (black) image in RGB format
        canvas = Image.new("RGB", (1024, 1024), color=(0, 0, 0))  # Black background in RGB

        # Calculate the top-left corner where the Canny image should be pasted
        top_left_x = center_x - (canny_width // 2)
        top_left_y = center_y - (canny_height // 2)

        # Paste the Canny image onto the canvas at the calculated coordinates
        canvas.paste(canny_image_pil, (top_left_x, top_left_y))

        # Convert back to tensor (in RGB format)
        result_tensor = pil2tensor(canvas)

        # Ensure the tensor is moved to the correct device
        result_tensor = result_tensor.to(comfy.model_management.get_torch_device())

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
    "Place Canny On Canvas": PlaceCannyOnCanvas
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Place Canny On Canvas": "Place Canny On Centered Canvas"
}
