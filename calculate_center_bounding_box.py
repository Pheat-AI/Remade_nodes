import comfy.utils
import numpy as np
import torch

class MaskBoundingBox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "padding": ("INT", { "default": 0, "min": 0, "max": 4096, "step": 1, }),
                "blur": ("INT", { "default": 0, "min": 0, "max": 256, "step": 1, }),
            },
            "optional": {
                "image_optional": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("MASK", "IMAGE", "x", "y", "width", "height", "center_x", "center_y")
    FUNCTION = "execute"
    CATEGORY = "REMADE/mask"

    def execute(self, mask, padding, blur, image_optional=None):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        if image_optional is None:
            image_optional = mask.unsqueeze(3).repeat(1, 1, 1, 3)

        # Resize the image if it's not the same size as the mask
        if image_optional.shape[1:] != mask.shape[1:]:
            image_optional = comfy.utils.common_upscale(image_optional.permute([0, 3, 1, 2]), mask.shape[2], mask.shape[1], upscale_method='bicubic', crop='center').permute([0, 2, 3, 1])

        # Match batch size
        if image_optional.shape[0] < mask.shape[0]:
            image_optional = torch.cat((image_optional, image_optional[-1].unsqueeze(0).repeat(mask.shape[0] - image_optional.shape[0], 1, 1, 1)), dim=0)
        elif image_optional.shape[0] > mask.shape[0]:
            image_optional = image_optional[:mask.shape[0]]

        # Blur the mask
        if blur > 0:
            if blur % 2 == 0:
                blur += 1
            mask = T.functional.gaussian_blur(mask.unsqueeze(1), blur).squeeze(1)

        # Calculate bounding box coordinates
        _, y, x = torch.where(mask)
        x1 = max(0, x.min().item() - padding)
        x2 = min(mask.shape[2], x.max().item() + 1 + padding)
        y1 = max(0, y.min().item() - padding)
        y2 = min(mask.shape[1], y.max().item() + 1 + padding)

        # Crop the mask and image to the bounding box
        mask = mask[:, y1:y2, x1:x2]
        image_optional = image_optional[:, y1:y2, x1:x2, :]

        # Calculate the center coordinates
        center_x = x1 + (x2 - x1) // 2
        center_y = y1 + (y2 - y1) // 2

        return (mask, image_optional, x1, y1, x2 - x1, y2 - y1, center_x, center_y)

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "MaskBoundingBox": MaskBoundingBox
}

# A dictionary that contains the friendly/human-readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskBoundingBox": "Mask Bounding Box with Center"
}
