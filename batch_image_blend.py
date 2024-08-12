import torch
import numpy as np
from PIL import Image, ImageOps

# IMAGE BLEND MASK NODE

class REMADE_Batch_Image_Blend_Mask:
    def __init__(self):
        pass 

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gen_images": ("IMAGE",),
                "segmented_image": ("IMAGE",),
                "segmented_image_mask": ("IMAGE",)
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_blend_mask"

    CATEGORY = "REMADE/Image"

    def image_blend_mask(self, gen_images, segmented_image, segmented_image_mask):
        # List of tensors
        blended_images = []

        print(f"gen_images: {gen_images.shape}, segmented_image: {segmented_image.shape}, segmented_image_mask: {segmented_image_mask.shape}")

        # Conversion to PIL format and converting mask to greyscale
        segmented_image = tensor2pil(segmented_image)
        segmented_image_mask = ImageOps.invert(tensor2pil(segmented_image_mask).convert('L'))

        for gen_image in gen_images:
            gen_image = tensor2pil(gen_image)
            print(f"gen_image: {gen_image.size}, segmented_image: {segmented_image.size}, segmented_image_mask: {segmented_image_mask.size}")

            masked_img = Image.composite(gen_image, segmented_image, segmented_image_mask.resize(gen_image.size))

            # Blend image
            blend_mask = Image.new(mode="L", size=gen_image.size, color=255)
            blend_mask = ImageOps.invert(blend_mask)
            img_result = Image.composite(gen_image, masked_img, blend_mask)

            result_tensor = pil2tensor(img_result)  
            print(f"result_tensor: {result_tensor.shape}")

            blended_images.append(result_tensor)
            del gen_image, masked_img, blend_mask

        # Convert blended_images array to tensor for use in Save node
        # Goes from array of tensors to (n, h, w, 3)
        # Not sure if we want this!!!
        result = torch.cat(blended_images, dim=0)
 

        return (result, )

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Batch Image Blend by Mask": REMADE_Batch_Image_Blend_Mask
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Batch Image Blend by Mask": "Batch Image Blend by Mask"
}
