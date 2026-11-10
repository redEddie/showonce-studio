"""Calibration transforms and RGB-D snapshot I/O, preserving original units."""
import json
import pathlib

import numpy as np
from scipy.spatial.transform import Rotation

from ..config import ROOT
CONFIG = pathlib.Path(__file__).parent / "data"
RUNS = ROOT / "runs"
CALIB = ROOT / "calib"


def load_json(path):
    return json.loads(pathlib.Path(path).read_text())


def save_json(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def make_T(R=np.eye(3), t=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def quat_wxyz(R):
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z])
    return q if q[0] >= 0 else -q


def rpy_deg(R):
    return Rotation.from_matrix(R).as_euler("xyz", degrees=True)


def latest_run():
    # capture.py가 만든 시각 이름(YYYYMMDD-HHMMSS)만 본다. lego-smoke 같은 이름은 정렬상 뒤에 와서 잘못 골린다
    runs = sorted(p for p in RUNS.glob("[0-9]" * 8 + "-" + "[0-9]" * 6) if p.is_dir())
    if not runs:
        raise SystemExit(f"런 디렉터리가 없다: {RUNS}. 먼저 calibrate-camera를 실행한다.")
    return runs[-1]


def load_frame(run):
    """capture.py가 저장한 RGB(uint8 HxWx3, RGB 순서), depth(m, float32), K(3x3), meta."""
    import cv2

    run = pathlib.Path(run)
    rgb = cv2.cvtColor(cv2.imread(str(run / "color" / "0.png")), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(run / "depth" / "0.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    K = np.loadtxt(run / "cam_K.txt")
    meta = load_json(run / "meta.json")
    return rgb, depth, K, meta


def backproject(depth, K, mask=None):
    """depth(m) → 카메라 광학 좌표(x 오른쪽, y 아래, z 앞) 점들. 깊이 0은 버린다."""
    v, u = np.nonzero((depth > 0) if mask is None else (mask & (depth > 0)))
    z = depth[v, u]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1)


def transform(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]
