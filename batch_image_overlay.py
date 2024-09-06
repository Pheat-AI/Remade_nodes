import torch
import numpy as np
from PIL import Image

# IMAGE OVERLAY NODE

class REMADE_Batch_Image_Overlay:
    def __init__(self):
        pass 

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gen_images": ("IMAGE",),
                "original_image": ("IMAGE",)
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_overlay"

    CATEGORY = "REMADE/Image"

    def image_overlay(self, gen_images, original_image):
        # List of tensors
        overlaid_images = []

        print(f"gen_images: {gen_images.shape}, original_image: {original_image.shape}")

        # Convert original image to PIL format
        original_image = tensor2pil(original_image)

        for gen_image in gen_images:
            # Convert each generated image to PIL
            gen_image = tensor2pil(gen_image)
            print(f"gen_image: {gen_image.size}, original_image: {original_image.size}")

            # Overlay original image onto the generated image
            # Assuming that we want to resize the original image to match the generated image size
            resized_original = original_image.resize(gen_image.size)
            overlaid_img = Image.alpha_composite(gen_image.convert("RGBA"), resized_original.convert("RGBA"))

            # Convert the result back to a tensor
            result_tensor = pil2tensor(overlaid_img)  
            print(f"result_tensor: {result_tensor.shape}")

            overlaid_images.append(result_tensor)
            del gen_image, overlaid_img

        # Convert overlaid_images array to tensor for use in Save node
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
    "Batch Image Overlay": REMADE_Batch_Image_Overlay
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Batch Image Overlay": "Batch Image Overlay"
}
