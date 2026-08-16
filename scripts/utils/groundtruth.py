import numpy as np

from utils.wsg84 import geodetic_to_ecef, get_enu_matrix


class GroundtruthProcessor:
    def __init__(self, origin: tuple[float, float, float]):
        self.origin = geodetic_to_ecef(*origin)
        self.R = get_enu_matrix(*origin)

    def __call__(
        self, latitude: float, longitude: float, altitude: float, covariance: np.ndarray
    ) -> tuple[tuple[float, float, float], np.ndarray]:
        # ECEF <- Geodetic
        pos = geodetic_to_ecef(latitude, longitude, altitude)
        # ENU@origin <- ECEF
        pos = self.R @ (pos - self.origin)

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

        return tuple(pos), covariance
