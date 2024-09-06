import torch
import numpy as np
from PIL import Image, ImageOps

# BATCH ENLARGED OVERLAY NODE

class REMADE_Batch_Enlarged_Overlay:
    def __init__(self):
        pass 

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gen_images": ("IMAGE",),  # A batch of generated images
                "original_image": ("IMAGE",),  # A single original image
                "enlarge_factor": ("FLOAT", {
                    "default": 1.05,  # Default enlarge factor
                    "min": 1.0, 
                    "max": 2.0, 
                    "step": 0.01,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch_enlarged_overlay"

    CATEGORY = "REMADE/Image"

    def batch_enlarged_overlay(self, gen_images, original_image, enlarge_factor):
        # List to hold the result tensors for each image in the batch
        overlaid_images = []

        # Convert the original image to PIL format and apply the enlargement
        original_image = tensor2pil(original_image).convert("RGBA")
        width, height = original_image.size
        new_size = (int(width * enlarge_factor), int(height * enlarge_factor))

        # Resize the original image (slightly enlarge)
        enlarged_original = original_image.resize(new_size, Image.LANCZOS)

        # Center the enlarged original image on a 1024x1024 canvas
        enlarged_image_on_canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))  # Transparent background
        offset_x = (width - new_size[0]) // 2
        offset_y = (height - new_size[1]) // 2
        enlarged_image_on_canvas.paste(enlarged_original, (offset_x, offset_y))

        # Iterate through each generated image in the batch
        for gen_image in gen_images:
            gen_image = tensor2pil(gen_image).convert("RGBA")

            # Overlay the enlarged original image onto the generated image
            result_image = Image.alpha_composite(gen_image, enlarged_image_on_canvas)

            # Convert the result back to tensor
            result_tensor = pil2tensor(result_image)

            # Append the resulting tensor to the batch result list
            overlaid_images.append(result_tensor)
        
        # Convert the list of tensors into a single batch tensor
        result = torch.cat(overlaid_images, dim=0)

        return (result, )

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Batch Enlarged Overlay": REMADE_Batch_Enlarged_Overlay
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Batch Enlarged Overlay": "Batch Enlarged Overlay"
}
