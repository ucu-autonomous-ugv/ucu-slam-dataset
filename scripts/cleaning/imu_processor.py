from rosbags.typesys.store import Typestore

from cleaning.proxy_writer import ProxyWriter
from cleaning.timestamp_processor import TimestampProcessor


class ImuProcessor:
    def __init__(
        self,
        timestamp_processor: TimestampProcessor,
        frame_id: str,
        topic: str,
        typestore: Typestore,
        proxy_writer: ProxyWriter | None = None,
    ):
        self.typestore = typestore
        self.frame_id = frame_id
        self.topic = topic
        self.timestamp_processor = timestamp_processor
        self.proxy_writer = proxy_writer

    def __call__(self, msg) -> None:
        msg.header.frame_id = self.frame_id
        msg.header.stamp, timestamp = self.timestamp_processor(msg.header.stamp)
        self.proxy_writer.write(self.topic, timestamp, msg, "sensor_msgs/msg/Imu")

    def close(self) -> None:
        pass
