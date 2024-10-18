import torch
from PIL import Image, ImageOps
import numpy as np

class REMADE_Batch_Image_Blend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_a": ("IMAGE",),
                "images_b": ("IMAGE",),
                "blend_percentage": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "batch_image_blend"

    CATEGORY = "REMADE/Image"

    def batch_image_blend(self, images_a, images_b, blend_percentage):
        print(f"Input shapes: images_a {images_a.shape}, images_b {images_b.shape}")
        print(f"Blend percentage: {blend_percentage}")

        assert images_a.shape[0] == images_b.shape[0], "Number of images in set A must match number of images in set B"
        
        blended_images = []

        for i in range(images_a.shape[0]):
            img_a = images_a[i]
            img_b = images_b[i]
            print(f"Processing image pair {i+1}")
            print(f"  Image A shape: {img_a.shape}")
            print(f"  Image B shape: {img_b.shape}")

            # Convert images to PIL
            pil_a = tensor2pil(img_a)
            pil_b = tensor2pil(img_b)

            print(f"  PIL Image sizes: A {pil_a.size}, B {pil_b.size}")

            # Blend image
            blend_mask = Image.new(mode="L", size=pil_a.size,
                                   color=(round(blend_percentage * 255)))
            blend_mask = ImageOps.invert(blend_mask)
            img_result = Image.composite(pil_a, pil_b, blend_mask)

            print(f"  Blended image size: {img_result.size}")

            # Convert back to tensor and append
            blended_tensor = pil2tensor(img_result)
            print(f"  Blended tensor shape: {blended_tensor.shape}")

            blended_images.append(blended_tensor)

            del pil_a, pil_b, blend_mask, img_result

        # Stack all blended images into a single tensor
        result = torch.stack(blended_images, dim=0)
        print(f"Final result shape: {result.shape}")

        return (result,)

def tensor2pil(image):
    return Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "REMADE Batch Image Blend": REMADE_Batch_Image_Blend
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "REMADE Batch Image Blend": "REMADE Batch Image Blend"
}
