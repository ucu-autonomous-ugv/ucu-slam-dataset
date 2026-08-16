from typing import List

import numpy as np
import yaml


def opencv_matrix_representer(dumper, data):
    return dumper.represent_mapping(
        "tag:yaml.org,2002:opencv-matrix",
        {
            "rows": data.shape[0],
            "cols": data.shape[1],
            "dt": "d",  # Assuming double precision
            "data": data.flatten().tolist(),
        },
    )


def opencv_matrix_constructor(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    rows = mapping["rows"]
    cols = mapping["cols"]
    data = np.array(mapping["data"], dtype=np.float64).reshape((rows, cols))
    return data


def register_opencv_matrix_yaml() -> None:
    yaml.add_representer(np.ndarray, opencv_matrix_representer)
    yaml.add_constructor("tag:yaml.org,2002:opencv-matrix", opencv_matrix_constructor)


def quaternion_to_rotation_matrix(quaternion: List[float]) -> np.ndarray:
    x, y, z, w = quaternion

    # Normalize quaternion
    norm = np.sqrt(x**2 + y**2 + z**2 + w**2)
    if norm == 0:
        raise ValueError("Cannot convert zero quaternion to rotation matrix")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    # Convert to rotation matrix
    matrix = np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ],
        dtype=np.float64,
    )

    return matrix


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> List[float]:
    if matrix.shape != (3, 3):
        raise ValueError(f"Rotation matrix must be 3x3, got {matrix.shape}")

    trace = np.trace(matrix)

    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (matrix[2, 1] - matrix[1, 2]) / S
        y = (matrix[0, 2] - matrix[2, 0]) / S
        z = (matrix[1, 0] - matrix[0, 1]) / S
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        S = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
        w = (matrix[2, 1] - matrix[1, 2]) / S
        x = 0.25 * S
        y = (matrix[0, 1] + matrix[1, 0]) / S
        z = (matrix[0, 2] + matrix[2, 0]) / S
    elif matrix[1, 1] > matrix[2, 2]:
        S = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
        w = (matrix[0, 2] - matrix[2, 0]) / S
        x = (matrix[0, 1] + matrix[1, 0]) / S
        y = 0.25 * S
        z = (matrix[1, 2] + matrix[2, 1]) / S
    else:
        S = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
        w = (matrix[1, 0] - matrix[0, 1]) / S
        x = (matrix[0, 2] + matrix[2, 0]) / S
        y = (matrix[1, 2] + matrix[2, 1]) / S
        z = 0.25 * S

    return np.array([x, y, z, w], dtype=np.float64)
