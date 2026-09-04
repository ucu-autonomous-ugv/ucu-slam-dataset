import sys

import numpy as np
from PIL import Image
from rosbags.typesys.store import Typestore


def msg_to_array(msg, is_bgr: bool = False) -> np.ndarray:
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

    if is_bgr and encoding == "rgb8":
        arr = arr[:, :, ::-1]

    if is_bgr and encoding == "rgba8":
        arr = arr[:, :, [2, 1, 0, 3]]

    if not is_bgr and encoding == "bgr8":
        arr = arr[:, :, ::-1]

    return arr


def array_to_msg(
    arr: np.ndarray, encoding: str, typestore: Typestore, is_bgr: bool = False
) -> "Image":
    Image = typestore.types["sensor_msgs/msg/Image"]

    h = arr.shape[0]
    w = arr.shape[1]

    assert arr.ndim == 3, "For now, only 3-channel images are supported"
    assert arr.shape[2] == 3, "For now, only 3-channel images are supported"

    if is_bgr and encoding == "rgb8":
        arr = arr[:, :, ::-1]
    elif not is_bgr and encoding == "bgr8":
        arr = arr[:, :, ::-1]

    if encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"Unsupported image encoding: {encoding}")

    data = arr.flatten()

    is_bigendian = False
    if arr.dtype.byteorder == "<":
        is_bigendian = True
    elif arr.dtype.byteorder == "=" and sys.byteorder == "big":
        is_bigendian = True

    return Image(
        header=None,
        height=h,
        width=w,
        encoding=encoding,
        is_bigendian=is_bigendian,
        step=w * arr.shape[2],
        data=data,
    )


def msg_to_image(msg) -> Image:
    rgb_arr = msg_to_array(msg, is_bgr=False)

    return Image.fromarray(rgb_arr)
