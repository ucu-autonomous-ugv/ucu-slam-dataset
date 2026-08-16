import argparse
import shutil
from pathlib import Path

import yaml
from rosbags.rosbag2 import Reader, StoragePlugin, Writer
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.store import Typestore
from tqdm import tqdm

from cleaning.camera_processor import CameraProcessor
from cleaning.gnss_processor import GnssProcessor
from cleaning.imu_processor import ImuProcessor
from cleaning.odometry_processor import OdometryProcessor
from cleaning.proxy_writer import ProxyWriter
from cleaning.tf_static_processor import TfStaticProcessor
from cleaning.timestamp_processor import TimestampProcessor
from utils.groundtruth import GroundtruthProcessor
from utils.transformations import register_opencv_matrix_yaml

register_opencv_matrix_yaml()


def process(
    input_bag_path: Path,
    output_bag_path: Path,
    typestore: Typestore,
    origin: tuple[float, float, float] = (49.81754, 24.02295, 378.3),
    output_format: StoragePlugin = StoragePlugin.MCAP,
    output_version: int = 9,
    calibration_file: Path = Path("./calibration/calibration.yaml"),
) -> None:
    with calibration_file.open(mode="r") as file:
        calibration_data = yaml.full_load(file)

    marked = set()

    frame_names = {
        "imu": "imu",
        "color": "color",
        "depth": "depth",
        "base": "base_link",
        "gnss": "gnss",
        "map": "map",
        "odom": "odom",
    }
    transformations = calibration_data["transformations"]

    topic_processors = {
        "/camera/camera/depth/image_rect_raw": CameraProcessor(
            timestamp_processor=TimestampProcessor(
                transformations["dt_base_color"] + transformations["dt_color_depth"],
                typestore,
            ),
            frame_id=frame_names["depth"],
            topic="/cam_depth/image_rect_raw",
            topic_camera_info="/cam_depth/camera_info",
            camera_info=calibration_data["cameras"]["depth"],
            typestore=typestore,
        ),
        "/camera/camera/aligned_depth_to_color/image_raw": CameraProcessor(
            timestamp_processor=TimestampProcessor(
                transformations["dt_base_color"] + transformations["dt_color_depth"],
                typestore,
            ),
            frame_id=frame_names["color"],
            topic="/cam_depth_aligned/image_raw",
            topic_camera_info="/cam_depth_aligned/camera_info",
            camera_info=calibration_data["cameras"]["depth_aligned"],
            typestore=typestore,
        ),
        "/camera/camera/color/image_raw": CameraProcessor(
            timestamp_processor=TimestampProcessor(
                transformations["dt_base_color"], typestore
            ),
            frame_id=frame_names["color"],
            topic="/cam_color/image_raw",
            topic_camera_info="/cam_color/camera_info",
            camera_info=calibration_data["cameras"]["color"],
            typestore=typestore,
        ),
        "/fix": GnssProcessor(
            groundtruth_processor=GroundtruthProcessor(origin),
            timestamp_processor=TimestampProcessor(
                transformations["dt_base_gnss"], typestore
            ),
            frame_id=frame_names["gnss"],
            frame_map_id=frame_names["map"],
            topic_gnss="/gnss",
            topic_pose="/pose",
            topic_tf="/tf",
            typestore=typestore,
        ),
        "/imu/data_raw": ImuProcessor(
            timestamp_processor=TimestampProcessor(
                transformations["dt_base_imu"], typestore
            ),
            frame_id=frame_names["imu"],
            topic="/imu",
            typestore=typestore,
        ),
        "/platform/odom": OdometryProcessor(
            timestamp_processor=TimestampProcessor(0, typestore),
            odom_frame_id=frame_names["odom"],
            base_frame_id=frame_names["base"],
            topic="/odom",
            topic_tf="/tf",
            typestore=typestore,
        ),
    }

    tf_static_processor = TfStaticProcessor(
        transformations=[
            (
                transformations["T_base_color"],
                frame_names["base"],
                frame_names["color"],
            ),
            (
                transformations["T_color_depth"],
                frame_names["color"],
                frame_names["depth"],
            ),
            (transformations["T_base_imu"], frame_names["base"], frame_names["imu"]),
            (transformations["T_base_gnss"], frame_names["base"], frame_names["gnss"]),
        ],
        topic="/tf_static",
        typestore=typestore,
    )

    with Reader(input_bag_path) as reader, Writer(
        output_bag_path,
        version=output_version,
        storage_plugin=output_format,
    ) as writer:
        proxy_writer = ProxyWriter(writer, typestore)
        tf_static_processor(proxy_writer)

        for connection, _, rawdata in tqdm(
            reader.messages(), total=reader.message_count
        ):
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

            processor = topic_processors.get(connection.topic)
            if processor is None:
                continue  # Ignore topics without a processor

            marked.add(connection.topic)
            processor(proxy_writer, msg)

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
        "--calibration-file",
        type=Path,
        help="Path to the calibration file (if needed)",
        default=Path("./calibration/calibration.yaml"),
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
    calibration_file = args.calibration_file

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
            calibration_file=calibration_file,
        )
