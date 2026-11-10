"""Quaternion representation used by the existing object pose file contract."""
import numpy as np
from scipy.spatial.transform import Rotation


def quat_wxyz(R):
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z])
    return q if q[0] >= 0 else -q
