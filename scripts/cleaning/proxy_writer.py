from rosbags.rosbag2 import Writer
from rosbags.typesys.store import Typestore


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

    def close(self) -> None:
        pass
