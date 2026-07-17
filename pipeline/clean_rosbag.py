import argparse
import shutil
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Reader, StoragePlugin, Writer
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.store import Typestore
from tqdm import tqdm
from utils.wsg84 import geodetic_to_ecef, get_enu_matrix


class ProxyWriter:
    def __init__(self, writer: Writer, typestore: Typestore):
        self.writer = writer
        self.connections = {}
        self.typestore = typestore

    def write(self, topic, timestamp, msg, msgtype):
        if topic not in self.connections:
            connection = self.writer.add_connection(
                topic, msgtype, typestore=self.typestore
            )
            self.connections[topic] = connection
        elif self.connections[topic].msgtype != msgtype:
            raise ValueError(
                f"Message type mismatch for topic '{topic}': "
                f"expected '{self.connections[topic].msgtype}', got '{msgtype}'"
            )

        connection = self.connections[topic]

        self.writer.write(
            connection, timestamp, self.typestore.serialize_cdr(msg, msgtype)
        )


def process_depth_image_aligned(proxy_writer: ProxyWriter, timestamp: int, msg) -> None:
    msg.header.frame_id = "camera_color"
    proxy_writer.write(
        "/cam_depth/image_aligned_raw", timestamp, msg, "sensor_msgs/msg/Image"
    )


def process_depth_image(proxy_writer: ProxyWriter, timestamp: int, msg) -> None:
    msg.header.frame_id = "camera_depth"
    proxy_writer.write("/cam_depth/image_raw", timestamp, msg, "sensor_msgs/msg/Image")


def process_color_image(proxy_writer: ProxyWriter, timestamp: int, msg) -> None:
    msg.header.frame_id = "camera_color"
    proxy_writer.write("/cam_color/image_raw", timestamp, msg, "sensor_msgs/msg/Image")


class GpsFixProcessor:
    def __init__(self, origin: tuple[float, float, float], typestore: Typestore):
        self.origin = geodetic_to_ecef(*origin)
        self.R = get_enu_matrix(*origin)
        self.typestore = typestore

    def __call__(self, proxy_writer: ProxyWriter, timestamp: int, msg) -> None:
        PoseWithCovarianceStamped = proxy_writer.typestore.types[
            "geometry_msgs/msg/PoseWithCovarianceStamped"
        ]
        PoseWithCovariance = proxy_writer.typestore.types[
            "geometry_msgs/msg/PoseWithCovariance"
        ]
        Pose = proxy_writer.typestore.types["geometry_msgs/msg/Pose"]
        Point = proxy_writer.typestore.types["geometry_msgs/msg/Point"]
        Quaternion = proxy_writer.typestore.types["geometry_msgs/msg/Quaternion"]

        msg.header.frame_id = "gps"
        proxy_writer.write("/gps", timestamp, msg, "sensor_msgs/msg/NavSatFix")

        latitude = msg.latitude
        longitude = msg.longitude
        altitude = msg.altitude

        # ECEF <- Geodetic
        pos = geodetic_to_ecef(latitude, longitude, altitude)
        # ENU@origin <- ECEF
        pos = self.R @ (pos - self.origin)

        if msg.position_covariance_type == msg.COVARIANCE_TYPE_UNKNOWN:
            covariance = np.zeros((6, 6), dtype=np.float64)
        else:
            covariance = np.array(msg.position_covariance, dtype=np.float64).reshape(
                3, 3
            )

            # P = ENU@position <- ECEF
            P = get_enu_matrix(latitude, longitude, altitude)
            # R = ENU@origin <- ECEF
            # => T = ENU@origin <- ENU@position
            T = self.R @ P.T

            # Covariance transformation
            covariance = T @ covariance @ T.T

            # Extend for orientation (set to zero for now)
            covariance = np.pad(
                covariance, ((0, 3), (0, 3)), mode="constant", constant_values=0
            )

        covariance = covariance.flatten()
        msg_pose = PoseWithCovarianceStamped(
            header=msg.header,
            pose=PoseWithCovariance(
                pose=Pose(
                    position=Point(x=pos[0], y=pos[1], z=pos[2]),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                covariance=covariance,
            ),
        )
        proxy_writer.write(
            "/pose", timestamp, msg_pose, "geometry_msgs/msg/PoseWithCovarianceStamped"
        )


def process_imu_data(proxy_writer: ProxyWriter, timestamp: int, msg) -> None:
    msg.header.frame_id = "imu"
    proxy_writer.write("/imu", timestamp, msg, "sensor_msgs/msg/Imu")


def process(
    input_bag_path: Path,
    output_bag_path: Path,
    typestore: Typestore,
    origin: tuple[float, float, float] = (49.81754, 24.02295, 378.3),
    output_format: StoragePlugin = StoragePlugin.MCAP,
    output_version: int = 9,
) -> None:
    marked = set()

    topic_processors = {
        "/camera/camera/depth/image_rect_raw": process_depth_image,  # TODO: temporarily using rectified
        "/camera/camera/aligned_depth_to_color/image_raw": process_depth_image_aligned,
        "/camera/camera/color/image_raw": process_color_image,
        "/gps/fix": GpsFixProcessor(origin, typestore),
        "/imu/data_raw": process_imu_data,
        # "/odometer/filtered": process_odom_data,  # TODO: implement odometry processing
    }

    with Reader(input_bag_path) as reader, Writer(
        output_bag_path,
        version=output_version,
        storage_plugin=output_format,
    ) as writer:
        proxy_writer = ProxyWriter(writer, typestore)

        for connection, timestamp, rawdata in tqdm(
            reader.messages(), total=reader.message_count
        ):
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

            processor = topic_processors.get(connection.topic)
            if processor is None:
                continue  # Ignore topics without a processor

            marked.add(connection.topic)
            processor(proxy_writer, timestamp, msg)

    for topic in topic_processors:
        if topic not in marked:
            print(f"Warning: No messages processed for topic '{topic}'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process ROS2 bag files")
    parser.add_argument("input_bag_path", type=Path, help="Path to the input bag file")
    parser.add_argument(
        "output_bag_path", type=Path, help="Path to the output bag file"
    )
    parser.add_argument(
        "--origin",
        nargs=3,
        type=float,
        default=(49.81754, 24.02295, 378.3),
        metavar=("LATITUDE", "LONGITUDE", "ALTITUDE"),
        help="Origin point for ENU conversion (latitude, longitude, altitude)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="mcap",
        choices=["mcap", "sqlite3"],
        help="Output format for the processed bag file",
    )
    parser.add_argument(
        "--output-version",
        type=int,
        default=9,
        choices=[8, 9],
        help="Output version for the processed bag file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output bag file if it already exists",
    )
    args = parser.parse_args()

    input_bag_path = args.input_bag_path
    output_bag_path = args.output_bag_path
    origin = tuple(args.origin)
    typestore = get_typestore(
        Stores.ROS2_JAZZY
    )  # Hardcoded to ROS2_JAZZY for now, adjust as needed
    output_format = {
        "mcap": StoragePlugin.MCAP,
        "sqlite3": StoragePlugin.SQLITE3,
    }[args.output_format]
    output_version = args.output_version

    if output_bag_path.exists() and not args.force:
        print(
            f"Error: Output bag file '{output_bag_path}' already exists. Please remove it or choose a different path."
        )
    else:
        shutil.rmtree(
            output_bag_path, ignore_errors=True
        )  # Remove existing output bag if it exists

        process(
            input_bag_path=input_bag_path,
            output_bag_path=output_bag_path,
            typestore=typestore,
            origin=origin,
            output_format=output_format,
            output_version=output_version,
        )
