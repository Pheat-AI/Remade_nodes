import torch

class CannyImageCropper:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "canny_image": ("IMAGE",),  # Canny image input
                "x": ("INT",),  # x-coordinate of the bounding box
                "y": ("INT",),  # y-coordinate of the bounding box
                "width": ("INT",),  # Width of the bounding box
                "height": ("INT",),  # Height of the bounding box
                "padding": ("INT", { "default": 5, "min": 0, "max": 100, "step": 1 }),  # Padding to add to the bounding box
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cropped_canny_image",)
    FUNCTION = "crop_canny_image"

    CATEGORY = "REMADE/Image"

    def crop_canny_image(self, canny_image, x, y, width, height, padding):
        # Apply padding to the bounding box to expand the cropping area
        x -= padding
        y -= padding
        width += 2 * padding
        height += 2 * padding

        # Ensure that the crop area doesn't exceed the image boundaries
        x = max(0, x)
        y = max(0, y)
        x2 = min(canny_image.shape[2], x + width)
        y2 = min(canny_image.shape[1], y + height)

        # Crop the canny image based on the expanded bounding box coordinates
        cropped_canny_image = canny_image[:, y:y2, x:x2, :]

        # Return the cropped canny image
        return (cropped_canny_image,)

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "CannyImageCropper": CannyImageCropper
}

# A dictionary that contains the friendly/human-readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "CannyImageCropper": "Canny Image Cropper"
}
