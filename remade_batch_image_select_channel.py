import torch
import numpy as np
from PIL import Image

class REMADE_Batch_Image_Select_Channel:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "channel": (['red', 'green', 'blue'],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "select_channel"

    CATEGORY = "REMADE/Image"

    def select_channel(self, images, channel='red'):
        # List of tensors
        processed_images = []

        for image in images:
            pil_image = tensor2pil(image)
            processed_image = self.convert_to_single_channel(pil_image, channel)
            processed_tensor = pil2tensor(processed_image)
            processed_images.append(processed_tensor)

        # Convert processed_images array to tensor
        result = torch.cat(processed_images, dim=0)

        return (result,)

    def convert_to_single_channel(self, image, channel='red'):
        # Convert to RGB mode to access individual channels
        image = image.convert('RGB')

        # Extract the desired channel and convert to greyscale
        if channel == 'red':
            channel_img = image.split()[0].convert('L')
        elif channel == 'green':
            channel_img = image.split()[1].convert('L')
        elif channel == 'blue':
            channel_img = image.split()[2].convert('L')
        else:
            raise ValueError(
                "Invalid channel option. Please choose 'red', 'green', or 'blue'.")

        # Convert the greyscale channel back to RGB mode
        channel_img = Image.merge(
            'RGB', (channel_img, channel_img, channel_img))

        return channel_img

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "REMADE Batch Image Select Channel": REMADE_Batch_Image_Select_Channel
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "REMADE Batch Image Select Channel": "REMADE Batch Image Select Channel"
}
