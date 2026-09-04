import numpy as np
from rosbags.typesys.store import Typestore

from cleaning.proxy_writer import ProxyWriter
from utils.transformations import rotation_matrix_to_quaternion


def publish_tf_static_transforms(
    transformations: list[
        tuple[np.ndarray, str, str]
    ],  # [(matrix, parent_frame_id, child_frame_id)]
    topic: str,
    typestore: Typestore,
    proxy_writer: ProxyWriter | None = None,
) -> None:
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]
    TFMessage = typestore.types["tf2_msgs/msg/TFMessage"]
    TransformStamped = typestore.types["geometry_msgs/msg/TransformStamped"]
    Transform = typestore.types["geometry_msgs/msg/Transform"]
    Vector3 = typestore.types["geometry_msgs/msg/Vector3"]
    Quaternion = typestore.types["geometry_msgs/msg/Quaternion"]
    proxy_writer = proxy_writer

    transforms = []

    for matrix, parent_frame_id, child_frame_id in transformations:
        tx, ty, tz = matrix[0:3, 3]
        rotation_matrix = matrix[0:3, 0:3]
        qx, qy, qz, qw = rotation_matrix_to_quaternion(rotation_matrix)

        transforms.append(
            TransformStamped(
                header=Header(stamp=Time(sec=0, nanosec=0), frame_id=parent_frame_id),
                child_frame_id=child_frame_id,
                transform=Transform(
                    translation=Vector3(x=float(tx), y=float(ty), z=float(tz)),
                    rotation=Quaternion(
                        x=float(qx), y=float(qy), z=float(qz), w=float(qw)
                    ),
                ),
            )
        )

    msg = TFMessage(transforms=transforms)
    proxy_writer.write(topic, 0, msg, "tf2_msgs/msg/TFMessage")
