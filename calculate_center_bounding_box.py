class CalculateCenterFromBoundingBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("INT",),
                "y": ("INT",),
                "width": ("INT",),
                "height": ("INT",),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("center_x", "center_y")
    FUNCTION = "calculate_center"
    CATEGORY = "essentials/calculations"

    def calculate_center(self, x, y, width, height):
        # Calculate the center coordinates
        center_x = x + (width // 2)
        center_y = y + (height // 2)
        
        # Return the center coordinates
        return (center_x, center_y)
