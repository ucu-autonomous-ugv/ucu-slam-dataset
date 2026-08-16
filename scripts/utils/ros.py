from typing import Generator

from rosbags.rosbag2 import Reader, Writer
from rosbags.typesys.store import Typestore

from scripts.utils.messages import MESSAGES, Message


class ROSReader:
    def __init__(self, input_bag_path: str, typestore: Typestore):
        self.reader = Reader(input_bag_path)
        self.typestore = typestore

    def open(self):
        self.reader.open()

    def close(self):
        self.reader.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def deserialize(self, rawdata, msgtype, timestamp):
        if msgtype not in MESSAGES:
            return None

        return MESSAGES[msgtype].deserialize(rawdata, timestamp, self.typestore)

    @property
    def message_count(self) -> int:
        return self.reader.message_count

    def messages(self) -> Generator[tuple[str, Message], None, None]:
        for connection, timestamp, rawdata in self.reader.messages():
            msgtype = connection.msgtype
            message = self.deserialize(rawdata, msgtype, timestamp)
            if message is not None:
                yield connection.topic, message


class ROSWriter:
    def __init__(
        self,
        output_bag_path: str,
        output_version: str,
        output_format: str,
        typestore: Typestore,
    ):
        self.writer = Writer(
            output_bag_path,
            version=output_version,
            storage_plugin=output_format,
        )
        self.connections = {}
        self.typestore = typestore

    def open(self):
        self.writer.open()

    def close(self):
        self.writer.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write(self, topic, message):
        if topic not in self.connections:
            connection = self.writer.add_connection(
                topic, message.msgtype, typestore=self.typestore
            )
            self.connections[topic] = connection
        elif self.connections[topic].msgtype != message.msgtype:
            raise ValueError(
                f"Message type mismatch for topic '{topic}': "
                f"expected '{self.connections[topic].msgtype}', got '{message.msgtype}'"
            )

        connection = self.connections[topic]

        self.writer.write(
            connection, message.timestamp, message.serialize(self.typestore)
        )
