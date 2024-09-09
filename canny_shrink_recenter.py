import torch
import numpy as np
from PIL import Image, ImageOps

class REMADE_Canny_Shrink_Recenter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "canny_image": ("IMAGE",),
                "mask": ("MASK",),
                "shrink_factor": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.01
                }),
                "padding": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 256,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("cropped_canny_image", "shrunken_canny_image", "recentered_canny_image", "center_x", "center_y")
    FUNCTION = "process_canny"

    CATEGORY = "REMADE/Image"

    def process_canny(self, canny_image, mask, shrink_factor, padding):
        # Convert the mask and canny_image to PIL for processing
        mask_pil = tensor2pil(mask).convert('L')
        canny_pil = tensor2pil(canny_image).convert('L')

        # Step 1: Find the bounding box for the mask (non-zero area)
        bbox = mask_pil.getbbox()
        if bbox is None:
            raise ValueError("No non-zero pixels found in mask")

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # Calculate center of the mask
        center_x = x1 + width // 2
        center_y = y1 + height // 2

        # Step 2: Crop the Canny image using the mask's bounding box + padding
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(1024, x2 + padding)
        y2_pad = min(1024, y2 + padding)

        cropped_canny = canny_pil.crop((x1_pad, y1_pad, x2_pad, y2_pad))

        # Step 3: Shrink the cropped Canny image by the shrink_factor
        shrunk_width = int(cropped_canny.size[0] * shrink_factor)
        shrunk_height = int(cropped_canny.size[1] * shrink_factor)
        shrunken_canny = cropped_canny.resize((shrunk_width, shrunk_height), Image.LANCZOS)

        # Step 4: Create a new 1024x1024 blank image
        recentered_canny = Image.new("L", (1024, 1024), 0)

        # Step 5: Place the shrunken Canny image onto the new blank image at the center of the mask
        paste_x = center_x - shrunk_width // 2
        paste_y = center_y - shrunk_height // 2
        recentered_canny.paste(shrunken_canny, (paste_x, paste_y))

        # Convert the results back to tensors
        cropped_canny_tensor = pil2tensor(cropped_canny)
        shrunken_canny_tensor = pil2tensor(shrunken_canny)
        recentered_canny_tensor = pil2tensor(recentered_canny)

        return cropped_canny_tensor, shrunken_canny_tensor, recentered_canny_tensor, center_x, center_y

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "Canny Shrink Recenter": REMADE_Canny_Shrink_Recenter
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Canny Shrink Recenter": "Canny Shrink and Recenter"
}
