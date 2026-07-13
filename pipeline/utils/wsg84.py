import numpy as np

# WSG84 model
A = 6378137.0  # semi-major axis in meters
F = 1 / 298.257223563  # flattening
E2 = F * (2 - F)


def geodetic_to_ecef(lat, lon, alt):
    phi = np.radians(lat)
    psi = np.radians(lon)
    h = alt
    n = A / (1 - E2 * (np.sin(phi)) ** 2) ** 0.5

    x = (n + h) * np.cos(phi) * np.cos(psi)
    y = (n + h) * np.cos(phi) * np.sin(psi)
    z = (n * (1 - E2) + h) * np.sin(phi)

    return np.array([x, y, z], dtype=np.float64)


def get_enu_matrix(lat, lon, alt):
    phi = np.radians(lat)
    psi = np.radians(lon)

    R = np.array(
        [
            [-np.sin(psi), np.cos(psi), 0],
            [-np.sin(phi) * np.cos(psi), -np.sin(phi) * np.sin(psi), np.cos(phi)],
            [np.cos(phi) * np.cos(psi), np.cos(phi) * np.sin(psi), np.sin(phi)],
        ],
        dtype=np.float64,
    )

    return R
