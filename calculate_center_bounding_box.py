class CalculateCenterFromBoundingBox:
    def __init__(self):
        pass 

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

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "Calculate Center From Bounding Box": CalculateCenterFromBoundingBox
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Calculate Center From Bounding Box": "Calculate Center from Bounding Box"
}
