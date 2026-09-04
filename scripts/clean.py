import argparse
import shutil
from pathlib import Path

import yaml
from rosbags.rosbag2 import Reader, StoragePlugin, Writer
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.store import Typestore
from tqdm import tqdm

from cleaning.blur_processor import BlurProcessor
from cleaning.camera_processor import CameraProcessor
from cleaning.gnss_processor import GnssProcessor
from cleaning.imu_processor import ImuProcessor
from cleaning.odometry_processor import OdometryProcessor
from cleaning.proxy_writer import ProxyWriter
from cleaning.tf_static_publisher import publish_tf_static_transforms
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
    calibration_file_path: Path | None = None,
    face_model_path: Path | None = None,
    license_plate_model_path: Path | None = None,
    device: str = "cpu",
    face_threshold: float = 0.67416,
    license_plate_threshold: float = 0.74475,
    nms_iou_threshold: float = 0.5,
    scale_factor_detections: float = 1.15,
    batch_size: int = 8,
    skip_seconds: float = 0.0,
    max_seconds: float | None = None,
) -> None:
    if calibration_file_path is not None:
        with calibration_file_path.open(mode="r") as file:
            calibration_data = yaml.full_load(file)

        transformations = calibration_data["transformations"]
        camera_infos = calibration_data["camera_infos"]
        should_publish_tf_static = True
    else:
        transformations = {
            "dt_base_color": 0,
            "dt_color_depth": 0,
            "dt_base_imu": 0,
            "dt_base_gnss": 0,
        }
        camera_infos = {
            "color": None,
            "depth": None,
            "depth_aligned": None,
        }
        should_publish_tf_static = False

    marked_topics = set()

    frame_names = {
        "imu": "imu",
        "color": "color",
        "depth": "depth",
        "base": "base_link",
        "gnss": "gnss",
        "map": "map",
        "odom": "odom",
    }

    blur_processor = None
    if face_model_path is not None and license_plate_model_path is not None:
        blur_processor = BlurProcessor(
            face_model_path=face_model_path,
            license_plate_model_path=license_plate_model_path,
            device=device,
            face_threshold=face_threshold,
            license_plate_threshold=license_plate_threshold,
            nms_iou_threshold=nms_iou_threshold,
            scale_factor_detections=scale_factor_detections,
            typestore=typestore,
        )
    elif face_model_path is not None or license_plate_model_path is not None:
        raise ValueError(
            "Both face_model_path and license_plate_model_path must be provided together."
        )

    if skip_seconds > 0 or max_seconds is not None:
        initial_timestamp = None
        with Reader(input_bag_path) as reader:
            for connection, _, rawdata in reader.messages():
                # Our skipping logic is based on IMU messages (granularity of 200 Hz)
                if connection.topic != "/imu/data_raw":
                    continue

                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

                initial_timestamp = (
                    msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                )
                break
            else:
                print(
                    "Warning: No IMU messages found in the bag. Skipping will not be applied."
                )
    else:
        initial_timestamp = 0

    min_timestamp = None
    max_timestamp = None

    with Reader(input_bag_path) as reader, Writer(
        output_bag_path,
        version=output_version,
        storage_plugin=output_format,
    ) as writer:
        proxy_writer = ProxyWriter(writer, typestore)

        topic_processors = {
            "/camera/camera/depth/image_rect_raw": CameraProcessor(
                timestamp_processor=TimestampProcessor(
                    transformations["dt_base_color"]
                    + transformations["dt_color_depth"],
                    typestore,
                ),
                frame_id=frame_names["depth"],
                topic="/cam_depth/image_rect_raw",
                topic_camera_info="/cam_depth/camera_info",
                camera_info=camera_infos["depth"],
                typestore=typestore,
                proxy_writer=proxy_writer,
            ),
            "/camera/camera/aligned_depth_to_color/image_raw": CameraProcessor(
                timestamp_processor=TimestampProcessor(
                    transformations["dt_base_color"]
                    + transformations["dt_color_depth"],
                    typestore,
                ),
                frame_id=frame_names["color"],
                topic="/cam_depth_aligned/image_raw",
                topic_camera_info="/cam_depth_aligned/camera_info",
                camera_info=camera_infos["depth_aligned"],
                typestore=typestore,
                proxy_writer=proxy_writer,
            ),
            "/camera/camera/color/image_raw": CameraProcessor(
                timestamp_processor=TimestampProcessor(
                    transformations["dt_base_color"], typestore
                ),
                frame_id=frame_names["color"],
                topic="/cam_color/image_raw",
                topic_camera_info="/cam_color/camera_info",
                camera_info=camera_infos["color"],
                typestore=typestore,
                blur_processor=blur_processor,
                proxy_writer=proxy_writer,
                blur_batch_size=batch_size,
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
                proxy_writer=proxy_writer,
            ),
            "/imu/data_raw": ImuProcessor(
                timestamp_processor=TimestampProcessor(
                    transformations["dt_base_imu"], typestore
                ),
                frame_id=frame_names["imu"],
                topic="/imu",
                typestore=typestore,
                proxy_writer=proxy_writer,
            ),
            "/platform/odom": OdometryProcessor(
                timestamp_processor=TimestampProcessor(0, typestore),
                odom_frame_id=frame_names["odom"],
                base_frame_id=frame_names["base"],
                topic="/odom",
                topic_tf="/tf",
                typestore=typestore,
                proxy_writer=proxy_writer,
            ),
        }

        if should_publish_tf_static:
            publish_tf_static_transforms(
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
                    (
                        transformations["T_base_imu"],
                        frame_names["base"],
                        frame_names["imu"],
                    ),
                    (
                        transformations["T_base_gnss"],
                        frame_names["base"],
                        frame_names["gnss"],
                    ),
                ],
                topic="/tf_static",
                typestore=typestore,
                proxy_writer=proxy_writer,
            )

        for connection, _, rawdata in tqdm(
            reader.messages(), total=reader.message_count
        ):
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

            # Note: assume all messages have a header with a timestamp. If not, this will raise an AttributeError.
            # if pipeline needs expansion, you should create separate function that will either choose header timestamp
            # or use record timestamp, depending on the message type.
            msg_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if msg_timestamp - initial_timestamp < skip_seconds:
                continue  # Skip messages that are within the skip_seconds window

            if (
                max_seconds is not None
                and msg_timestamp - initial_timestamp > max_seconds + skip_seconds
            ):
                continue  # Skip messages that exceed the max_seconds limit

            processor = topic_processors.get(connection.topic)
            if processor is None:
                continue  # Ignore topics without a processor

            max_timestamp = (
                msg_timestamp
                if max_timestamp is None
                else max(max_timestamp, msg_timestamp)
            )
            min_timestamp = (
                msg_timestamp
                if min_timestamp is None
                else min(min_timestamp, msg_timestamp)
            )

            marked_topics.add(connection.topic)
            processor(msg)

        for processor in topic_processors.values():
            processor.close()

    for topic in topic_processors:
        if topic not in marked_topics:
            print(f"Warning: No messages processed for topic '{topic}'")

    print(f"Processing complete. Output bag file saved to '{output_bag_path}'")
    print(f"Processed messages timestamp range: {min_timestamp} to {max_timestamp}")
    print(
        f"Total duration of processed messages: {max_timestamp - min_timestamp} seconds"
    )


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
        default=None,
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
    parser.add_argument(
        "--face-model-path",
        type=Path,
        default=None,
        help="Path to the face detection model",
    )
    parser.add_argument(
        "--license-plate-model-path",
        type=Path,
        default=None,
        help="Path to the license plate detection model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run the detection models on (cpu or cuda)",
    )
    parser.add_argument(
        "--face-threshold",
        type=float,
        default=0.67416,
        help="Confidence threshold for face detection",
    )
    parser.add_argument(
        "--license-plate-threshold",
        type=float,
        default=0.74475,
        help="Confidence threshold for license plate detection",
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=0.5,
        help="Non-maximum suppression IoU threshold for detections",
    )
    parser.add_argument(
        "--scale-factor-detections",
        type=float,
        default=1.15,
        help="Scale factor for resizing images before detection",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing images in the blur processor",
    )
    parser.add_argument(
        "--skip-seconds",
        type=float,
        default=0.0,
        help="Number of seconds to skip from the start of the bag file (based on IMU timestamps)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Maximum number of seconds to process from the start of the bag file (based on IMU timestamps)",
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
    calibration_file_path = args.calibration_file

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
            calibration_file_path=calibration_file_path,
            face_model_path=args.face_model_path,
            license_plate_model_path=args.license_plate_model_path,
            device=args.device,
            face_threshold=args.face_threshold,
            license_plate_threshold=args.license_plate_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            scale_factor_detections=args.scale_factor_detections,
            batch_size=args.batch_size,
            skip_seconds=args.skip_seconds,
            max_seconds=args.max_seconds,
        )
