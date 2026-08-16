import numpy as np
from PIL import Image


def msg_to_image(msg) -> Image:
    h = int(msg.height)
    w = int(msg.width)
    encoding = msg.encoding.lower()

    if encoding in ("rgb8", "bgr8", "rgba8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        channels = 3 if encoding in ("rgb8", "bgr8") else 4

    elif encoding in ("16uc1",):
        dtype = ">u2" if msg.is_bigendian else "<u2"
        arr = np.frombuffer(msg.data, dtype=dtype).astype(np.uint16)
        channels = 1

    else:
        raise ValueError(f"Unsupported image encoding: {encoding}")

    assert (
        arr.size == h * w * channels
    ), f"Data size {arr.size} does not match expected size {h * w * channels}"

    arr = arr.reshape((h, w, channels)) if channels > 1 else arr.reshape((h, w))

    if encoding == "bgr8":
        arr = arr[:, :, ::-1]

    return Image.fromarray(arr)
