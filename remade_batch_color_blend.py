import torch
import numpy as np
import cv2
from typing import Literal

# Constants for RGB to YIQ conversion
R, G, B = 0.299, 0.587, 0.114

class REMADE_Batch_Color_Blend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blend_images": ("IMAGE",),
                "base_images": ("IMAGE",),
                "mode": (["Hue", "Saturation", "Color", "Luminosity"],)
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend"
    CATEGORY = "REMADE/Image"

    def blend(self, blend_images: torch.Tensor, base_images: torch.Tensor, mode: Literal["Hue", "Saturation", "Color", "Luminosity"]):
        print(f"Input shapes: blend_images {blend_images.shape}, base_images {base_images.shape}")

        # List to store blended images
        blended_images = []

        # Ensure we have the same number of blend and base images
        assert blend_images.shape[0] == base_images.shape[0], "Number of blend images must match number of base images"

        for i, (blend_img, base_img) in enumerate(zip(blend_images, base_images)):
            print(f"Processing image pair {i+1}")
            print(f"  Blend image shape: {blend_img.shape}")
            print(f"  Base image shape: {base_img.shape}")

            # Convert tensors to cv2 format
            cv_blend_img = tensor2cv(blend_img)
            cv_base_img = tensor2cv(base_img)

            print(f"  CV2 image shapes: blend {cv_blend_img.shape}, base {cv_base_img.shape}")

            # Perform color blending
            blended_cv_img = color_blend(base_image=cv_base_img, blend_image=cv_blend_img, mode=mode)

            print(f"  Blended CV2 image shape: {blended_cv_img.shape}")

            # Convert back to tensor and append to list
            blended_tensor = cv2tensor(blended_cv_img)
            print(f"  Blended tensor shape: {blended_tensor.shape}")

            blended_images.append(blended_tensor)

        # Stack all blended images into a single tensor
        result = torch.cat(blended_images, dim=0)
        print(f"Final result shape: {result.shape}")

        # Ensure the result is in the format expected by ComfyUI (B, H, W, C)
        result = result.permute(0, 2, 3, 1)
        print(f"Transformed result shape: {result.shape}")

        return (result,)

def color_blend(base_image, blend_image, mode: Literal["Hue", "Saturation", "Color", "Luminosity"]):
    """
    Args:
        base_image (cv2)
        blend_image (cv2)
    Return:
        cv2
    """
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB).astype(np.float32)/255
    blend_image = cv2.cvtColor(blend_image, cv2.COLOR_BGR2RGB).astype(np.float32)/255

    # Convert to HSY color space
    hsy_base = rgb2hsy(base_image)
    hsy_blend = rgb2hsy(blend_image)

    if mode=="Hue":
        hsy_out = np.stack([hsy_blend[:,:,0], hsy_base[:,:,1], hsy_base[:,:,2]], axis=-1)
    elif mode=="Saturation":
        hsy_out = np.stack([hsy_base[:,:,0], hsy_blend[:,:,1], hsy_base[:,:,2]], axis=-1)
    elif mode=="Color":
        hsy_out = np.stack([hsy_blend[:,:,0], hsy_blend[:,:,1], hsy_base[:,:,2]], axis=-1)
    elif mode=="Luminosity":
        hsy_out = np.stack([hsy_base[:,:,0], hsy_base[:,:,1], hsy_blend[:,:,2]], axis=-1)
    else: assert False, f"{mode} is not a valid mode"
    rgb_out = hsy2rgb(hsy_out)
    return cv2.cvtColor((rgb_out*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def rgb2hsy(image):
    """image is normalized to [0,1]
    """
    image = np.minimum(np.maximum(image, 0), 1)
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]

    h = np.zeros_like(r)
    s = np.zeros_like(g)
    y = R * r + G * g + B * b
    
    mask_gray = (r == g) & (g == b)
    s[mask_gray] = 0
    h[mask_gray] = 0

    mask_sector_0 = ((r >= g) & (g >= b)) & ~mask_gray # Sector 0: 0° - 60°
    s[mask_sector_0] = r[mask_sector_0] - b[mask_sector_0]
    h[mask_sector_0] = 60 * (g[mask_sector_0] - b[mask_sector_0]) / s[mask_sector_0]

    mask_sector_1 = (g > r) & (r >= b)
    s[mask_sector_1] = g[mask_sector_1] - b[mask_sector_1]
    h[mask_sector_1] = 60 * (g[mask_sector_1] - r[mask_sector_1]) / s[mask_sector_1]  + 60

    mask_sector_2 = (g >= b) & (b > r)
    s[mask_sector_2] = g[mask_sector_2] - r[mask_sector_2]
    h[mask_sector_2] = 60 * (b[mask_sector_2] - r[mask_sector_2]) / s[mask_sector_2] + 120

    mask_sector_3 = (b > g) & (g > r)
    s[mask_sector_3] = b[mask_sector_3] - r[mask_sector_3]
    h[mask_sector_3] = 60 * (b[mask_sector_3] - g[mask_sector_3]) / s[mask_sector_3] + 180

    mask_sector_4 = (b > r) & (r >= g)
    s[mask_sector_4] = b[mask_sector_4] - g[mask_sector_4]
    h[mask_sector_4] = 60 * (r[mask_sector_4] - g[mask_sector_4]) / s[mask_sector_4] + 240

    mask_sector_5 = ~(mask_gray | mask_sector_0 | mask_sector_1 | mask_sector_2 | mask_sector_3 | mask_sector_4)
    s[mask_sector_5] = r[mask_sector_5] - g[mask_sector_5]
    h[mask_sector_5] = 60 * (r[mask_sector_5] - b[mask_sector_5]) / s[mask_sector_5] + 300
    
    hsy = np.zeros_like(image)
    hsy[:,:,0] = h % 360
    hsy[:,:,1] = np.minimum(np.maximum(s, 0), 1)
    hsy[:,:,2] = np.minimum(np.maximum(y, 0), 1)
    return hsy
 
def hsy2rgb(image):
    h, s, y = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    h = h % 360
    s = np.minimum(np.maximum(s, 0), 1)
    y = np.minimum(np.maximum(y, 0), 1)
    
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)
    k = np.zeros_like(h)

    mask_sector_0 = (h >= 0) & (h < 60)
    k[mask_sector_0] = s[mask_sector_0] * h[mask_sector_0] / 60
    b[mask_sector_0] = y[mask_sector_0] - R * s[mask_sector_0] - G * k[mask_sector_0]
    r[mask_sector_0] = b[mask_sector_0] + s[mask_sector_0]
    g[mask_sector_0] = b[mask_sector_0] + k[mask_sector_0]

    mask_sector_1 = (h >= 60) & (h < 120)
    k[mask_sector_1] = s[mask_sector_1] * (h[mask_sector_1] - 60) / 60
    g[mask_sector_1] = y[mask_sector_1] + B * s[mask_sector_1] + R * k[mask_sector_1]
    b[mask_sector_1] = g[mask_sector_1] - s[mask_sector_1]
    r[mask_sector_1] = g[mask_sector_1] - k[mask_sector_1]

    mask_sector_2 = (h >= 120) & (h < 180)
    k[mask_sector_2] = s[mask_sector_2] * (h[mask_sector_2] - 120) / 60
    r[mask_sector_2] = y[mask_sector_2] - G * s[mask_sector_2] - B * k[mask_sector_2]
    g[mask_sector_2] = r[mask_sector_2] + s[mask_sector_2]
    b[mask_sector_2] = r[mask_sector_2] + k[mask_sector_2]

    mask_sector_3 = (h >= 180) & (h < 240)
    k[mask_sector_3] = s[mask_sector_3] * (h[mask_sector_3] - 180) / 60
    b[mask_sector_3] = y[mask_sector_3] + R * s[mask_sector_3] + G * k[mask_sector_3]
    r[mask_sector_3] = b[mask_sector_3] - s[mask_sector_3]
    g[mask_sector_3] = b[mask_sector_3] - k[mask_sector_3]

    mask_sector_4 = (h >= 240) & (h < 300)
    k[mask_sector_4] = s[mask_sector_4] * (h[mask_sector_4] - 240) / 60
    g[mask_sector_4] = y[mask_sector_4] - B * s[mask_sector_4] - R * k[mask_sector_4]
    b[mask_sector_4] = g[mask_sector_4] + s[mask_sector_4]
    r[mask_sector_4] = g[mask_sector_4] + k[mask_sector_4]

    mask_sector_5 = h >= 300
    k[mask_sector_5] = s[mask_sector_5] * (h[mask_sector_5] - 300) / 60
    r[mask_sector_5] = y[mask_sector_5] + G * s[mask_sector_5] + B * k[mask_sector_5]
    g[mask_sector_5] = r[mask_sector_5] - s[mask_sector_5]
    b[mask_sector_5] = r[mask_sector_5] - k[mask_sector_5]
            
    return np.minimum(np.maximum(np.stack([r, g, b], axis=-1), 0), 1)

def tensor2cv(image):
    numpy_image = image.cpu().numpy().squeeze()
    if len(numpy_image.shape) == 2:
        numpy_image = np.stack([numpy_image] * 3, axis=-1)
    elif len(numpy_image.shape) == 3 and numpy_image.shape[0] == 3:
        numpy_image = numpy_image.transpose(1, 2, 0)
    return (numpy_image * 255).astype(np.uint8)

def cv2tensor(image):
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
    if len(tensor.shape) == 3:
        tensor = tensor.permute(2, 0, 1)
    return tensor.unsqueeze(0)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "REMADE Batch Color Blend": REMADE_Batch_Color_Blend
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "REMADE Batch Color Blend": "REMADE Batch Color Blend"
}
