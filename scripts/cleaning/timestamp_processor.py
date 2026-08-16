from rosbags.typesys.store import Typestore


class TimestampProcessor:
    def __init__(self, deltatime: float, typestore: Typestore):
        self.deltatime = deltatime
        self.typestore = typestore
        self.Time = self.typestore.types["builtin_interfaces/msg/Time"]

    def __call__(self, stamp) -> tuple:
        timestamp = stamp.sec * 1_000_000_000 + stamp.nanosec
        timestamp += int(self.deltatime * 1_000_000_000)
        sec, nanosec = divmod(timestamp, 1_000_000_000)

        return (
            self.Time(
                sec=int(sec),
                nanosec=int(nanosec),
            ),
            timestamp,
        )
