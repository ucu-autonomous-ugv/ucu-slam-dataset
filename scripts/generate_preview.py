import argparse
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import matplotlib
import numpy as np
from PIL import Image
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm

from utils.images import msg_to_image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TILE_SIZE = 256
MIN_ZOOM = 0
MAX_ZOOM = 19


@dataclass(frozen=True)
class GpsPoint:
    timestamp_ns: int
    latitude: float
    longitude: float
    altitude: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate preview PNGs from a cleaned ROS 2 bag."
    )
    parser.add_argument("input_bag_path", type=Path, help="Path to the cleaned bag")
    parser.add_argument(
        "--output-preview-path",
        type=Path,
        help="Path to save the median color frame preview PNG",
    )
    parser.add_argument(
        "--output-trajectory-path",
        type=Path,
        help="Path to save the GNSS trajectory preview PNG",
    )
    parser.add_argument(
        "--origin",
        nargs=3,
        type=float,
        default=(49.81754, 24.02295, 378.3),
        metavar=("LATITUDE", "LONGITUDE", "ALTITUDE"),
        help="Reference origin point shown on the map",
    )
    parser.add_argument(
        "--arrow-interval-sec",
        type=float,
        default=10.0,
        help="Time interval between motion arrows on the GNSS trajectory",
    )
    parser.add_argument(
        "--preview-at",
        type=float,
        default=0.5,
        help="Time from 0 to 1 to extract the preview frame (default: 0.5)",
    )
    return parser.parse_args()


def lat_to_mercator_normalized(lat: float) -> float:
    lat = max(min(lat, 85.05112878), -85.05112878)
    lat_rad = math.radians(lat)
    return 0.5 - (math.log(math.tan(math.pi / 4.0 + lat_rad / 2.0)) / (2.0 * math.pi))


def lon_to_tile_x(lon: float, zoom: int) -> float:
    n = 2**zoom
    return ((lon + 180.0) / 360.0) * n


def lat_to_tile_y(lat: float, zoom: int) -> float:
    n = 2**zoom
    return lat_to_mercator_normalized(lat) * n


def tile_x_to_lon(x: float, zoom: int) -> float:
    n = 2**zoom
    return (x / n) * 360.0 - 180.0


def tile_y_to_lat(y: float, zoom: int) -> float:
    n = 2**zoom
    y_norm = y / n
    merc_n = math.pi * (1.0 - 2.0 * y_norm)
    return math.degrees(math.atan(math.sinh(merc_n)))


def lon_to_mercator_x_m(lon: float) -> float:
    return 6378137.0 * math.radians(lon)


def lat_to_mercator_y_m(lat: float) -> float:
    lat = max(min(lat, 85.05112878), -85.05112878)
    return 6378137.0 * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))


def calculate_limits(
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    padding: float,
    ratio_width: float,
    ratio_height: float,
) -> tuple[float, float, float, float]:
    lon_span = max(max_lon - min_lon, 1e-4)
    lat_span = max(max_lat - min_lat, 1e-4)

    lon_pad = lon_span * padding
    lat_pad = lat_span * padding

    min_lon -= lon_pad
    max_lon += lon_pad
    min_lat -= lat_pad
    max_lat += lat_pad

    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat

    current_ratio = lon_span / lat_span
    target_ratio = ratio_width / ratio_height

    if current_ratio > target_ratio:
        target_lat_span = lon_span / target_ratio
        lat_center = (min_lat + max_lat) / 2.0
        min_lat = lat_center - target_lat_span / 2.0
        max_lat = lat_center + target_lat_span / 2.0
    else:
        target_lon_span = lat_span * target_ratio
        lon_center = (min_lon + max_lon) / 2.0
        min_lon = lon_center - target_lon_span / 2.0
        max_lon = lon_center + target_lon_span / 2.0

    return min_lon, max_lon, min_lat, max_lat


def choose_zoom(
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    target_px_width: float,
    target_px_height: float,
) -> int:
    x_span_norm = max((max_lon - min_lon) / 360.0, 1e-12)
    y_span_norm = max(
        abs(lat_to_mercator_normalized(max_lat) - lat_to_mercator_normalized(min_lat)),
        1e-12,
    )

    for zoom in range(MAX_ZOOM, MIN_ZOOM - 1, -1):
        world_px = TILE_SIZE * (2**zoom)
        span_px_x = x_span_norm * world_px
        span_px_y = y_span_norm * world_px
        if span_px_x <= target_px_width and span_px_y <= target_px_height:
            return zoom
    return MIN_ZOOM


def iter_tile_indices(
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    zoom: int,
):
    x0 = int(math.floor(lon_to_tile_x(min_lon, zoom)))
    x1 = int(math.floor(lon_to_tile_x(max_lon, zoom)))
    y0 = int(math.floor(lat_to_tile_y(max_lat, zoom)))
    y1 = int(math.floor(lat_to_tile_y(min_lat, zoom)))

    n = 2**zoom
    for x in range(x0, x1 + 1):
        if not (0 <= x < n):
            continue
        for y in range(y0, y1 + 1):
            if 0 <= y < n:
                yield x, y


def fetch_tile_image(zoom: int, x: int, y: int):
    url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
    request = Request(url, headers={"User-Agent": "husky-eda-preview/1.0"})
    with urlopen(request, timeout=15) as response:
        data = response.read()
    return plt.imread(BytesIO(data), format="png")


def decode_image(msg) -> Image.Image:
    try:
        return msg_to_image(msg)
    except Exception:
        pass

    encoding = msg.encoding.lower()
    h = int(msg.height)
    w = int(msg.width)

    if encoding in ("rgb8", "bgr8", "rgba8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        channels = 3 if encoding in ("rgb8", "bgr8") else 4
        expected = h * w * channels
        if arr.size < expected:
            arr = np.resize(arr, expected)
        arr = arr[:expected].reshape((h, w, channels))
        if encoding == "bgr8":
            arr = arr[:, :, ::-1]
        if encoding == "rgba8":
            arr = arr[:, :, :3]
        return Image.fromarray(arr)

    if encoding in ("mono8", "8uc1"):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        expected = h * w
        if arr.size < expected:
            arr = np.resize(arr, expected)
        return Image.fromarray(arr[:expected].reshape((h, w))).convert("RGB")

    if encoding in ("16uc1",):
        dtype = ">u2" if getattr(msg, "is_bigendian", 0) else "<u2"
        arr = np.frombuffer(msg.data, dtype=dtype).astype(np.uint16)
        expected = h * w
        if arr.size < expected:
            arr = np.resize(arr, expected)
        arr = arr[:expected].reshape((h, w))
        arr8 = (arr >> 8).astype(np.uint8)
        return Image.fromarray(arr8).convert("RGB")

    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def collect_preview_data(input_bag_path: Path, typestore) -> tuple[int, list[GpsPoint]]:
    color_count = 0
    gnss_points: list[GpsPoint] = []

    with Reader(input_bag_path) as reader:
        for connection, timestamp, rawdata in tqdm(
            reader.messages(), total=reader.message_count
        ):
            if connection.topic not in {"/cam_color/image_raw", "/gnss"}:
                continue

            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

            if connection.topic == "/cam_color/image_raw":
                color_count += 1
            elif connection.topic == "/gnss":
                gnss_points.append(
                    GpsPoint(
                        timestamp_ns=int(timestamp),
                        latitude=float(msg.latitude),
                        longitude=float(msg.longitude),
                        altitude=float(getattr(msg, "altitude", 0.0)),
                    )
                )

    return color_count, gnss_points


def extract_median_color_image(
    input_bag_path: Path, typestore, color_index: int
) -> Image.Image:
    current = 0

    with Reader(input_bag_path) as reader:
        for connection, _, rawdata in tqdm(
            reader.messages(), total=reader.message_count
        ):
            if connection.topic != "/cam_color/image_raw":
                continue

            if current == color_index:
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                return decode_image(msg)
            current += 1

    raise RuntimeError("Median color frame was not found in the bag")


def draw_map_with_trajectory(
    points: list[GpsPoint],
    origin: tuple[float, float, float],
    output_path: Path,
    arrow_interval_sec: float,
) -> None:
    origin_lat, origin_lon, _ = origin
    origin_x = lon_to_mercator_x_m(origin_lon)
    origin_y = lat_to_mercator_y_m(origin_lat)

    xs = [lon_to_mercator_x_m(p.longitude) - origin_x for p in points]
    ys = [lat_to_mercator_y_m(p.latitude) - origin_y for p in points]

    point_min_x = min(min(xs), 0.0)
    point_max_x = max(max(xs), 0.0)
    point_min_y = min(min(ys), 0.0)
    point_max_y = max(max(ys), 0.0)

    lats = [p.latitude for p in points]
    lons = [p.longitude for p in points]

    min_lon, max_lon, min_lat, max_lat = calculate_limits(
        min(lons),
        max(lons),
        min(lats),
        max(lats),
        padding=0.1,
        ratio_width=4,
        ratio_height=3,
    )

    fig_width, fig_height, dpi = 10.0, 10.0, 160
    target_px_width = fig_width * dpi
    target_px_height = fig_height * dpi
    zoom = choose_zoom(
        min_lon,
        max_lon,
        min_lat,
        max_lat,
        target_px_width,
        target_px_height,
    )

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    tile_min_x = float("inf")
    tile_max_x = float("-inf")
    tile_min_y = float("inf")
    tile_max_y = float("-inf")
    tiles_drawn = 0

    for tile_x, tile_y in iter_tile_indices(min_lon, max_lon, min_lat, max_lat, zoom):
        try:
            image = fetch_tile_image(zoom, tile_x, tile_y)
        except URLError as exc:
            print(
                f"Warning: failed to fetch tile z={zoom} x={tile_x} y={tile_y}: {exc}"
            )
            continue

        lon_left = tile_x_to_lon(tile_x, zoom)
        lon_right = tile_x_to_lon(tile_x + 1, zoom)
        lat_top = tile_y_to_lat(tile_y, zoom)
        lat_bottom = tile_y_to_lat(tile_y + 1, zoom)

        x_left = lon_to_mercator_x_m(lon_left) - origin_x
        x_right = lon_to_mercator_x_m(lon_right) - origin_x
        y_top = lat_to_mercator_y_m(lat_top) - origin_y
        y_bottom = lat_to_mercator_y_m(lat_bottom) - origin_y

        ax.imshow(
            image,
            extent=(x_left, x_right, y_bottom, y_top),
            origin="upper",
            interpolation="bilinear",
            zorder=0,
        )
        tiles_drawn += 1
        tile_min_x = min(tile_min_x, x_left, x_right)
        tile_max_x = max(tile_max_x, x_left, x_right)
        tile_min_y = min(tile_min_y, y_bottom, y_top)
        tile_max_y = max(tile_max_y, y_bottom, y_top)

    if tiles_drawn == 0:
        print("Warning: no map tiles downloaded, exporting trajectory without basemap.")
        ax.set_facecolor("#f5f5f5")
        tile_min_x, tile_max_x = point_min_x, point_max_x
        tile_min_y, tile_max_y = point_min_y, point_max_y

    ax.plot(xs, ys, color="#d62828", linewidth=2.0, zorder=3, label="Trajectory")

    ax.scatter(
        [xs[0]], [ys[0]], marker="D", s=60, color="#2a9d8f", zorder=4, label="Start"
    )
    ax.scatter(
        [0.0], [0.0], marker="s", s=70, color="#457b9d", zorder=4, label="Origin"
    )
    ax.scatter(
        [xs[-1]], [ys[-1]], marker="X", s=80, color="#1d3557", zorder=4, label="End"
    )

    interval_ns = max(int(arrow_interval_sec * 1e9), 1)
    next_arrow_ts = points[0].timestamp_ns + interval_ns
    for i in range(len(points) - 1):
        current = points[i]
        nxt = points[i + 1]

        if current.timestamp_ns < next_arrow_ts <= nxt.timestamp_ns:
            ax.annotate(
                "",
                xy=(
                    lon_to_mercator_x_m(nxt.longitude) - origin_x,
                    lat_to_mercator_y_m(nxt.latitude) - origin_y,
                ),
                xytext=(
                    lon_to_mercator_x_m(current.longitude) - origin_x,
                    lat_to_mercator_y_m(current.latitude) - origin_y,
                ),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#1d3557",
                    lw=1.4,
                    shrinkA=0,
                    shrinkB=0,
                    mutation_scale=10,
                ),
                zorder=5,
            )
            next_arrow_ts += interval_ns

    ax.set_xlim(tile_min_x, tile_max_x)
    ax.set_ylim(tile_min_y, tile_max_y)
    ax.set_xlabel("Easting relative to origin (m)")
    ax.set_ylabel("Northing relative to origin (m)")
    ax.grid(False)
    ax.legend(loc="upper right")
    ax.set_aspect("equal", adjustable="box")

    fig.text(
        0.01,
        0.01,
        "Map data: OpenStreetMap contributors",
        fontsize=8,
        color="#444444",
        ha="left",
        va="bottom",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    args = parse_args()
    input_bag_path = args.input_bag_path
    origin = tuple(args.origin)
    typestore = get_typestore(Stores.ROS2_JAZZY)

    color_count, gnss_points = collect_preview_data(input_bag_path, typestore)
    gnss_points.sort(key=lambda p: p.timestamp_ns)

    if args.output_preview_path:
        if color_count == 0:
            raise RuntimeError(
                "No /cam_color/image_raw messages found in the cleaned bag"
            )

        image_index = round(color_count * args.preview_at)
        image = extract_median_color_image(
            input_bag_path=input_bag_path,
            typestore=typestore,
            color_index=image_index,
        )
        image.save(args.output_preview_path, format="PNG")
        print(f"Saved: {args.output_preview_path}")

    if args.output_trajectory_path:
        if not gnss_points:
            raise RuntimeError("No /gnss messages found in the cleaned bag")

        draw_map_with_trajectory(
            points=gnss_points,
            origin=origin,
            output_path=args.output_trajectory_path,
            arrow_interval_sec=max(args.arrow_interval_sec, 0.1),
        )
        print(f"Saved: {args.output_trajectory_path}")


if __name__ == "__main__":
    main()
