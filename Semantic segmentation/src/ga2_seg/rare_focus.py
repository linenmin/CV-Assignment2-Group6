from .labels import class_names_to_label_ids


RARE_FOCUS_CLASS_NAMES = (
    "bicycle",
    "chair",
    "cow",
    "sheep",
    "pottedplant",
)
RARE_FOCUS_CLASS_IDS = class_names_to_label_ids(RARE_FOCUS_CLASS_NAMES)
RARE_FOCUS_DUPLICATE_FACTOR = 2
RARE_FOCUS_PROBABILITY = 0.5
RARE_FOCUS_CAT_MAX_RATIO = 0.65
