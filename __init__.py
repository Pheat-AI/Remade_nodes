from .batch_image_blend import NODE_CLASS_MAPPINGS as BLEND_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as BLEND_DISPLAY_NAMES
from .batch_image_overlay import NODE_CLASS_MAPPINGS as OVERLAY_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as OVERLAY_DISPLAY_NAMES
from .remove_black_to_transparent import NODE_CLASS_MAPPINGS as REMOVE_BLACK_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as REMOVE_BLACK_DISPLAY_NAMES
from .batch_enlarged_overlay import NODE_CLASS_MAPPINGS as ENLARGED_OVERLAY_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as ENLARGED_OVERLAY_DISPLAY_NAMES
from .calculate_center_bounding_box import NODE_CLASS_MAPPINGS as CENTRE_BOUNDING_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as CENTRE_BOUNDING_DISPLAY_NAMES
from .canny_image_cropper import NODE_CLASS_MAPPINGS as CANNY_CROPPER_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as CANNY_CROPPER_DISPLAY_NAMES
from .shrink_cropped_canny import NODE_CLASS_MAPPINGS as SHRINK_CANNY_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as SHRINK_CANNY_DISPLAY_NAMES
from .place_shrunken_canny_on_canvas import NODE_CLASS_MAPPINGS as CANNY_CANVAS_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as CANNY_CANVAS_DISPLAY_NAMES

# Merge all node mappings into one
NODE_CLASS_MAPPINGS = {
    **BLEND_CLASS_MAPPINGS,
    **OVERLAY_CLASS_MAPPINGS,
    **REMOVE_BLACK_CLASS_MAPPINGS,
    **ENLARGED_OVERLAY_CLASS_MAPPINGS,
    **CENTRE_BOUNDING_CLASS_MAPPINGS,
    **CANNY_CROPPER_CLASS_MAPPINGS,
    **SHRINK_CANNY_CLASS_MAPPINGS,
    **CANNY_CANVAS_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **BLEND_DISPLAY_NAMES,
    **OVERLAY_DISPLAY_NAMES,
    **REMOVE_BLACK_DISPLAY_NAMES,
    **ENLARGED_OVERLAY_DISPLAY_NAMES,
    **CENTRE_BOUNDING_DISPLAY_NAMES,
    **CANNY_CROPPER_DISPLAY_NAMES,
    **SHRINK_CANNY_DISPLAY_NAMES,
    **CANNY_CANVAS_DISPLAY_NAMES,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']


