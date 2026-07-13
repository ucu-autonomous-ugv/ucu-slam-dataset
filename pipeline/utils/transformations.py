import numpy as np


def format_transformation(matrix: np.ndarray) -> dict:
    assert matrix.ndim == 2, f"Transformation matrix must be 2D"

    return {
        "cols": matrix.shape[1],
        "rows": matrix.shape[0],
        "data": matrix.flatten().tolist(),
    }
