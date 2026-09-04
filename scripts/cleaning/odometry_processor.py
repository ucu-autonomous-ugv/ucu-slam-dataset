from rosbags.typesys.store import Typestore

from cleaning.proxy_writer import ProxyWriter
from cleaning.timestamp_processor import TimestampProcessor


class OdometryProcessor:
    def __init__(
        self,
        timestamp_processor: TimestampProcessor,
        odom_frame_id: str,
        base_frame_id: str,
        topic: str,
        topic_tf: str,
        typestore: Typestore,
        proxy_writer: ProxyWriter | None = None,
    ):
        self.typestore = typestore
        self.odom_frame_id = odom_frame_id
        self.base_frame_id = base_frame_id
        self.topic = topic
        self.topic_tf = topic_tf
        self.timestamp_processor = timestamp_processor
        self.proxy_writer = proxy_writer

        self.Header = self.typestore.types["std_msgs/msg/Header"]
        self.Vector3 = self.typestore.types["geometry_msgs/msg/Vector3"]
        self.Transform = self.typestore.types["geometry_msgs/msg/Transform"]
        self.TransformStamped = self.typestore.types[
            "geometry_msgs/msg/TransformStamped"
        ]
        self.TFMessage = self.typestore.types["tf2_msgs/msg/TFMessage"]
        self.Quaternion = self.typestore.types["geometry_msgs/msg/Quaternion"]

    def __call__(self, msg) -> None:
        msg.header.frame_id = self.odom_frame_id
        msg.header.stamp, timestamp = self.timestamp_processor(msg.header.stamp)
        msg.child_frame_id = self.base_frame_id
        self.proxy_writer.write(self.topic, timestamp, msg, "nav_msgs/msg/Odometry")

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        tf_msg = self.TFMessage(
            transforms=[
                self.TransformStamped(
                    header=self.Header(
                        stamp=msg.header.stamp, frame_id=self.odom_frame_id
                    ),
                    child_frame_id=self.base_frame_id,
                    transform=self.Transform(
                        translation=self.Vector3(
                            x=float(p.x), y=float(p.y), z=float(p.z)
                        ),
                        rotation=self.Quaternion(
                            x=float(q.x), y=float(q.y), z=float(q.z), w=float(q.w)
                        ),
                    ),
                )
            ]
        )

        self.proxy_writer.write(
            self.topic_tf, timestamp, tf_msg, "tf2_msgs/msg/TFMessage"
        )

    def close(self) -> None:
        pass
