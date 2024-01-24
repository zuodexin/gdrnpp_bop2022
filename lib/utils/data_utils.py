import mmcv


def read_image_mmcv(file_name, format=None):
    """# NOTE modified from detectron2, use mmcv instead of PIL to read an
    image into the given format.

    Args:
        file_name (str): image file path
        format (str): "BGR" | "RGB" | "L" | "unchanged"
    Returns:
        image (np.ndarray): an HWC image
    """
    flag = "color"
    channel_order = "bgr"
    if format == "RGB":
        channel_order = "rgb"
    elif format == "L":
        flag = "grayscale"
    elif format == "unchanged":
        flag = "unchanged"
    else:
        if format not in [None, "BGR"]:
            raise ValueError(f"Invalid format: {format}")

    image = mmcv.imread(file_name, flag, channel_order)
    return image
