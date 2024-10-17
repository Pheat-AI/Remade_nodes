import torch
import comfy.utils

MAX_RESOLUTION = 8192

def composite(destination, source, x, y, mask = None, multiplier = 8, resize_source = False):
    source = source.to(destination.device)
    if resize_source:
        source = torch.nn.functional.interpolate(source, size=(destination.shape[2], destination.shape[3]), mode="bilinear")

    source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])

    x = max(-source.shape[3] * multiplier, min(x, destination.shape[3] * multiplier))
    y = max(-source.shape[2] * multiplier, min(y, destination.shape[2] * multiplier))

    left, top = (x // multiplier, y // multiplier)
    right, bottom = (left + source.shape[3], top + source.shape[2],)

    if mask is None:
        mask = torch.ones_like(source)
    else:
        mask = mask.to(destination.device, copy=True)
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(source.shape[2], source.shape[3]), mode="bilinear")
        mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])

    visible_width, visible_height = (destination.shape[3] - left + min(0, x), destination.shape[2] - top + min(0, y),)

    mask = mask[:, :, :visible_height, :visible_width]
    inverse_mask = torch.ones_like(mask) - mask

    source_portion = mask * source[:, :, :visible_height, :visible_width]
    destination_portion = inverse_mask  * destination[:, :, top:bottom, left:right]

    destination[:, :, top:bottom, left:right] = source_portion + destination_portion
    return destination

class REMADE_Batch_ImageCompositeMasked:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destinations": ("IMAGE",),
                "sources": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "y": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "resize_source": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch_composite"

    CATEGORY = "REMADE/Image"

    def batch_composite(self, destinations, sources, x, y, resize_source, mask=None):
        assert destinations.shape[0] == sources.shape[0], "Number of destination images must match number of source images"
        
        output_images = []

        for dest, src in zip(destinations, sources):
            dest = dest.unsqueeze(0)  # Add batch dimension
            src = src.unsqueeze(0)    # Add batch dimension
            
            dest = dest.clone().movedim(-1, 1)
            src = src.movedim(-1, 1)
            
            output = composite(dest, src, x, y, mask, 1, resize_source).movedim(1, -1)
            output_images.append(output.squeeze(0))  # Remove batch dimension
        
        return (torch.stack(output_images),)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "REMADE Batch Image Composite Masked": REMADE_Batch_ImageCompositeMasked
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "REMADE Batch Image Composite Masked": "REMADE Batch Image Composite Masked"
}
