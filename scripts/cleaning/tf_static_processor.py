import numpy as np
from rosbags.typesys.store import Typestore

from cleaning.proxy_writer import ProxyWriter
from utils.transformations import rotation_matrix_to_quaternion


class TfStaticProcessor:
    def __init__(
        self,
        transformations: list[
            tuple[np.ndarray, str, str]
        ],  # [(matrix, parent_frame_id, child_frame_id)]
        topic: str,
        typestore: Typestore,
    ):
        self.typestore = typestore
        self.topic = topic
        self.transformations = transformations

        self.Header = self.typestore.types["std_msgs/msg/Header"]
        self.Time = self.typestore.types["builtin_interfaces/msg/Time"]
        self.TFMessage = self.typestore.types["tf2_msgs/msg/TFMessage"]
        self.TransformStamped = self.typestore.types[
            "geometry_msgs/msg/TransformStamped"
        ]
        self.Transform = self.typestore.types["geometry_msgs/msg/Transform"]
        self.Vector3 = self.typestore.types["geometry_msgs/msg/Vector3"]
        self.Quaternion = self.typestore.types["geometry_msgs/msg/Quaternion"]

    def __call__(self, proxy_writer: ProxyWriter) -> None:
        transforms = []

        for matrix, parent_frame_id, child_frame_id in self.transformations:
            tx, ty, tz = matrix[0:3, 3]
            rotation_matrix = matrix[0:3, 0:3]
            qx, qy, qz, qw = rotation_matrix_to_quaternion(rotation_matrix)

            transforms.append(
                self.TransformStamped(
                    header=self.Header(
                        stamp=self.Time(sec=0, nanosec=0), frame_id=parent_frame_id
                    ),
                    child_frame_id=child_frame_id,
                    transform=self.Transform(
                        translation=self.Vector3(x=float(tx), y=float(ty), z=float(tz)),
                        rotation=self.Quaternion(
                            x=float(qx), y=float(qy), z=float(qz), w=float(qw)
                        ),
                    ),
                )
            )

        msg = self.TFMessage(transforms=transforms)
        proxy_writer.write(self.topic, 0, msg, "tf2_msgs/msg/TFMessage")
        self._published = True
