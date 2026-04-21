CLASS_NAMES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

BACKGROUND_LABEL = "background"
ALL_CLASSES = (BACKGROUND_LABEL, *CLASS_NAMES)
NUM_CLASSES_WITH_BACKGROUND = len(CLASS_NAMES) + 1


def build_palette(num_classes: int) -> list[tuple[int, int, int]]:
    return [
        (
            (class_id * 37) % 255,
            (class_id * 67) % 255,
            (class_id * 97) % 255,
        )
        for class_id in range(num_classes)
    ]


PALETTE = build_palette(NUM_CLASSES_WITH_BACKGROUND)
