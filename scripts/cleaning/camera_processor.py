from typing import Literal, NotRequired, TypedDict

import numpy as np
from rosbags.typesys.store import Typestore

from cleaning.blur_processor import BlurProcessor
from cleaning.proxy_writer import ProxyWriter
from cleaning.timestamp_processor import TimestampProcessor
from utils.images import array_to_msg, msg_to_array


class CameraInfoDict(TypedDict):
    resolution: tuple[int, int]
    camera_model: Literal["pinhole"]
    distortion_model: Literal["plumb_bob", "none"]
    intrinsics: tuple[float, float, float, float]
    distortion_coeffs: NotRequired[tuple[float, float, float, float]]
    fps: float
    depth_scale: NotRequired[float]


class CameraProcessor:
    def __init__(
        self,
        timestamp_processor: TimestampProcessor,
        frame_id: str,
        topic: str,
        typestore: Typestore,
        topic_camera_info: str | None = None,
        camera_info: CameraInfoDict | None = None,
        blur_processor: BlurProcessor | None = None,
        proxy_writer: ProxyWriter | None = None,
        blur_batch_size: int = 8,
    ):
        self.typestore = typestore
        self.frame_id = frame_id
        self.topic = topic
        self.topic_camera_info = topic_camera_info
        self.camera_info = camera_info
        self.timestamp_processor = timestamp_processor
        self.blur_processor = blur_processor
        self.proxy_writer = proxy_writer
        self.blur_batch_size = blur_batch_size
        self._pending_msgs: list = []

    def _write_msg(self, msg) -> None:
        CameraInfo = self.typestore.types["sensor_msgs/msg/CameraInfo"]
        RegionOfInterest = self.typestore.types["sensor_msgs/msg/RegionOfInterest"]

        msg.header.frame_id = self.frame_id
        msg.header.stamp, timestamp = self.timestamp_processor(msg.header.stamp)
        self.proxy_writer.write(self.topic, timestamp, msg, "sensor_msgs/msg/Image")

        if self.topic_camera_info is None or self.camera_info is None:
            return

        # This represents no ditortion (i.e. model == none)
        distortion_model = "plumb_bob"
        distortion_coeffs = [0.0, 0.0, 0.0, 0.0]

        if self.camera_info["distortion_model"] == "plumb_bob":
            distortion_model = "plumb_bob"
            distortion_coeffs = list(self.camera_info["distortion_coeffs"])

        msg_camera_info = CameraInfo(
            header=msg.header,
            height=self.camera_info["resolution"][1],
            width=self.camera_info["resolution"][0],
            distortion_model=distortion_model,
            d=np.asarray(distortion_coeffs, dtype=np.float64),
            k=np.asarray(
                [
                    self.camera_info["intrinsics"][0],
                    0.0,
                    self.camera_info["intrinsics"][2],
                    0.0,
                    self.camera_info["intrinsics"][1],
                    self.camera_info["intrinsics"][3],
                    0.0,
                    0.0,
                    1.0,
                ],
                dtype=np.float64,
            ),
            r=np.asarray(
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                dtype=np.float64,
            ),
            p=np.asarray(
                [
                    self.camera_info["intrinsics"][0],
                    0.0,
                    self.camera_info["intrinsics"][2],
                    0.0,
                    0.0,
                    self.camera_info["intrinsics"][1],
                    self.camera_info["intrinsics"][3],
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
                dtype=np.float64,
            ),
            binning_x=0,
            binning_y=0,
            roi=RegionOfInterest(
                x_offset=0,
                y_offset=0,
                height=self.camera_info["resolution"][1],
                width=self.camera_info["resolution"][0],
                do_rectify=False,
            ),
        )
        self.proxy_writer.write(
            self.topic_camera_info,
            timestamp,
            msg_camera_info,
            "sensor_msgs/msg/CameraInfo",
        )

    def _flush_pending(self) -> None:
        if not self._pending_msgs:
            return

        pending_msgs = self._pending_msgs
        self._pending_msgs = []

        if self.blur_processor is None:
            for msg in pending_msgs:
                self._write_msg(msg)
            return

        bgr_arrs = [msg_to_array(msg, is_bgr=True) for msg in pending_msgs]
        blurred_arrs = self.blur_processor.blur_batch(bgr_arrs)

        for original_msg, blurred_arr in zip(pending_msgs, blurred_arrs):
            blurred_msg = array_to_msg(
                blurred_arr,
                encoding=original_msg.encoding,
                typestore=self.typestore,
                is_bgr=True,
            )
            blurred_msg.header = original_msg.header
            self._write_msg(blurred_msg)

    def __call__(self, msg) -> None:
        if self.blur_processor is None:
            self._write_msg(msg)
            return

        self._pending_msgs.append(msg)
        if len(self._pending_msgs) >= self.blur_batch_size:
            self._flush_pending()

    def close(self) -> None:
        self._flush_pending()
