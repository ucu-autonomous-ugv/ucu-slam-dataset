import argparse
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.store import Typestore
from tqdm import tqdm

from utils.images import msg_to_image


def format_timestamp(timestamp_ns: int) -> float:
    timestamp_s = timestamp_ns / 1_000_000_000
    return f"{timestamp_s:.9f}"  # Format to 9 decimal places for nanosecond precision


class Processor(ABC):
    @abstractmethod
    def __call__(self, timestamp: int, msg) -> None:
        pass

    @abstractmethod
    def close(self):
        pass


class FileProcessor(Processor):
    def __init__(self, process_func, output_file_path: Path):
        self.process_func = process_func
        self.is_header_written = False
        self.output_file_path = output_file_path
        self.output_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(output_file_path, "w")

    @property
    def header(self) -> str | None:
        annotations = inspect.get_annotations(self.process_func)
        return_annotation = annotations.get("return", None)
        if return_annotation is not None and hasattr(return_annotation, "__metadata__"):
            return return_annotation.__metadata__[0]

        return None

    def __call__(self, timestamp: int, msg) -> None:
        line = self.process_func(timestamp, msg)

        if not self.is_header_written and self.header is not None:
            self.file.write(f"#{self.header}\n")

        self.is_header_written = True
        self.file.write(" ".join(map(str, line)) + "\n")

    def close(self):
        if not self.file.closed:
            self.file.close()


def process_pose(timestamp: int, msg) -> Annotated[list, "timestamp x y z qx qy qz qw"]:
    position = msg.pose.pose.position
    orientation = msg.pose.pose.orientation

    return [
        format_timestamp(timestamp),
        position.x,
        position.y,
        position.z,
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    ]


def process_imu(timestamp: int, msg) -> Annotated[list, "timestamp ax ay az gx gy gz"]:
    linear_acceleration = msg.linear_acceleration
    angular_velocity = msg.angular_velocity

    return [
        format_timestamp(timestamp),
        linear_acceleration.x,
        linear_acceleration.y,
        linear_acceleration.z,
        angular_velocity.x,
        angular_velocity.y,
        angular_velocity.z,
    ]


class ImageProcessor(Processor):
    def __init__(self, output_dir: Path, output_file_path: Path):
        self.timestamp_to_path = {}
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file_path = output_file_path
        self.output_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.header_written = False
        self.file = open(output_file_path, "w")

    def __call__(self, timestamp: int, msg) -> None:
        image = msg_to_image(msg)
        path = self.output_dir / (str(timestamp) + ".png")
        image.save(path, format="PNG")

        if not self.header_written:
            self.file.write("#timestamp path\n")
            self.header_written = True

        path_txt = path.relative_to(
            self.output_file_path.parent,
            walk_up=True,
        )  # Get relative path to the output directory
        self.timestamp_to_path[timestamp] = path_txt
        self.file.write(f"{format_timestamp(timestamp)} {path_txt}\n")

    def close(self):
        if not self.file.closed:
            self.file.close()


def process_associations(
    output_file_path: Path,
    color_processor: ImageProcessor,
    depth_processor: ImageProcessor,
) -> None:
    color_timestamps = set(color_processor.timestamp_to_path.keys())
    depth_timestamps = set(depth_processor.timestamp_to_path.keys())
    common_timestamps = color_timestamps.intersection(depth_timestamps)

    print(
        f"Found {len(common_timestamps)} common timestamps for color {len(color_timestamps)} color and {len(depth_timestamps)} depth images."
    )

    with open(output_file_path, "w") as file:
        file.write("#color_timestamp color_path depth_timestamp depth_path\n")
        for timestamp in sorted(common_timestamps):
            timestamp_string = format_timestamp(timestamp)
            file.write(
                f"{timestamp_string} {color_processor.timestamp_to_path[timestamp]} {timestamp_string} {depth_processor.timestamp_to_path[timestamp]}\n"
            )


def process(
    input_bag_path: Path,
    output_path: Path,
    typestore: Typestore,
) -> None:
    marked = set()

    topic_processors = {
        "/cam_depth/image_rect_raw": ImageProcessor(
            output_path / "depth_not_aligned", output_path / "depth_not_aligned.txt"
        ),
        "/cam_depth_aligned/image_raw": ImageProcessor(
            output_path / "depth", output_path / "depth.txt"
        ),
        "/cam_color/image_raw": ImageProcessor(
            output_path / "rgb", output_path / "rgb.txt"
        ),
        "/pose": FileProcessor(process_pose, output_path / "groundtruth.txt"),
        "/imu": FileProcessor(process_imu, output_path / "imu.txt"),
    }

    try:
        with Reader(input_bag_path) as reader:
            for connection, timestamp, rawdata in tqdm(
                reader.messages(), total=reader.message_count
            ):
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

                processor = topic_processors.get(connection.topic)
                if processor is None:
                    continue  # Ignore topics without a processor

                marked.add(connection.topic)
                processor(timestamp, msg)

        process_associations(
            output_path / "associations.txt",
            color_processor=topic_processors["/cam_color/image_raw"],
            depth_processor=topic_processors["/cam_depth_aligned/image_raw"],
        )

        process_associations(
            output_path / "associations_not_aligned.txt",
            color_processor=topic_processors["/cam_color/image_raw"],
            depth_processor=topic_processors["/cam_depth/image_rect_raw"],
        )
    finally:
        for processor in topic_processors.values():
            processor.close()

    for topic in topic_processors:
        if topic not in marked:
            print(f"Warning: No messages processed for topic '{topic}'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process ROS2 bag files")
    parser.add_argument("input_bag_path", type=Path, help="Path to the input bag file")
    parser.add_argument("output_path", type=Path, help="Path to the output directory")
    args = parser.parse_args()

    input_bag_path = args.input_bag_path
    output_path = args.output_path
    typestore = get_typestore(
        Stores.ROS2_JAZZY
    )  # Hardcoded to ROS2_JAZZY for now, adjust as needed

    if output_path.exists():
        print(
            f"Error: Output direcotry '{output_path}' already exists. Please remove it or choose a different path."
        )
    else:
        process(
            input_bag_path=input_bag_path,
            output_path=output_path,
            typestore=typestore,
        )
