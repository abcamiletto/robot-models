"""STL mesh loading."""

import struct
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int

_BINARY_TRIANGLE = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])


def load_stl_mesh(
    path: Path, *, scale: Float[np.ndarray, "3"]
) -> tuple[Float[np.ndarray, "V 3"], Int[np.ndarray, "F 3"]]:
    """Load an STL mesh as shared vertices and triangle faces, with an MJCF ``scale`` applied."""
    data = path.read_bytes()
    corners = _binary_corners(data) if _is_binary(data) else _ascii_corners(data.decode("utf-8"))
    vertices, faces = np.unique(corners, axis=0, return_inverse=True)
    faces = faces.reshape(-1, 3)
    # A mirroring scale flips the winding, so restore outward-facing triangles.
    if np.prod(scale) < 0:
        faces = faces[:, ::-1]
    return vertices * scale, faces


def _binary_corners(data: bytes) -> Float[np.ndarray, "C 3"]:
    count = struct.unpack_from("<I", data, 80)[0]
    triangles = np.frombuffer(data, dtype=_BINARY_TRIANGLE, count=count, offset=84)
    return triangles["vertices"].reshape(-1, 3).astype(np.float64)


def _ascii_corners(text: str) -> Float[np.ndarray, "C 3"]:
    corners = [line.split()[1:] for line in text.splitlines() if line.strip().startswith("vertex")]
    if not corners or len(corners) % 3:
        raise ValueError("ASCII STL contains no triangular facets")
    return np.asarray(corners, dtype=np.float64)


def _is_binary(data: bytes) -> bool:
    return len(data) >= 84 and 84 + struct.unpack_from("<I", data, 80)[0] * 50 == len(data)
