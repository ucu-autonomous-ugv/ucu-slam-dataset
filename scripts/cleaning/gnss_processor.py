import numpy as np
from rosbags.typesys.store import Typestore

from cleaning.proxy_writer import ProxyWriter
from cleaning.timestamp_processor import TimestampProcessor
from utils.groundtruth import GroundtruthProcessor


class GnssProcessor:
    def __init__(
        self,
        groundtruth_processor: GroundtruthProcessor,
        timestamp_processor: TimestampProcessor,
        frame_id: str,
        frame_map_id: str,
        topic_gnss: str,
        topic_pose: str,
        topic_tf: str,
        typestore: Typestore,
    ):
        self.typestore = typestore
        self.groundtruth_processor = groundtruth_processor
        self.timestamp_processor = timestamp_processor
        self.frame_id = frame_id
        self.frame_map_id = frame_map_id
        self.topic_gnss = topic_gnss
        self.topic_pose = topic_pose
        self.topic_tf = topic_tf

        self.Header = self.typestore.types["std_msgs/msg/Header"]
        self.PoseWithCovarianceStamped = self.typestore.types[
            "geometry_msgs/msg/PoseWithCovarianceStamped"
        ]
        self.PoseWithCovariance = self.typestore.types[
            "geometry_msgs/msg/PoseWithCovariance"
        ]
        self.Pose = self.typestore.types["geometry_msgs/msg/Pose"]
        self.Point = self.typestore.types["geometry_msgs/msg/Point"]
        self.Quaternion = self.typestore.types["geometry_msgs/msg/Quaternion"]
        self.Vector3 = self.typestore.types["geometry_msgs/msg/Vector3"]
        self.Transform = self.typestore.types["geometry_msgs/msg/Transform"]
        self.TransformStamped = self.typestore.types[
            "geometry_msgs/msg/TransformStamped"
        ]
        self.TFMessage = self.typestore.types["tf2_msgs/msg/TFMessage"]

    def __call__(self, proxy_writer: ProxyWriter, msg) -> None:
        msg.header.frame_id = self.frame_id
        msg.header.stamp, timestamp = self.timestamp_processor(msg.header.stamp)
        proxy_writer.write(self.topic_gnss, timestamp, msg, "sensor_msgs/msg/NavSatFix")

        latitude = msg.latitude
        longitude = msg.longitude
        altitude = msg.altitude

        if msg.position_covariance_type == msg.COVARIANCE_TYPE_UNKNOWN:
            covariance = np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64
            )
        else:
            covariance = np.array(msg.position_covariance, dtype=np.float64).reshape(
                3, 3
            )

        (x, y, z), covariance = self.groundtruth_processor(
            latitude, longitude, altitude, covariance
        )

        covariance = covariance.flatten()
        msg_pose = self.PoseWithCovarianceStamped(
            header=msg.header,
            pose=self.PoseWithCovariance(
                pose=self.Pose(
                    position=self.Point(x=x, y=y, z=z),
                    orientation=self.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                covariance=covariance,
            ),
        )
        proxy_writer.write(
            self.topic_pose,
            timestamp,
            msg_pose,
            "geometry_msgs/msg/PoseWithCovarianceStamped",
        )

        msg_transformation = self.TFMessage(
            transforms=[
                self.TransformStamped(
                    header=self.Header(
                        stamp=msg.header.stamp, frame_id=self.frame_map_id
                    ),
                    child_frame_id=self.frame_id,
                    transform=self.Transform(
                        translation=self.Vector3(x=x, y=y, z=z),
                        rotation=self.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                    ),
                )
            ]
        )
        proxy_writer.write(
            self.topic_tf, timestamp, msg_transformation, "tf2_msgs/msg/TFMessage"
        )
