"""Fixed-joint markerless registration of a saved RGB-D frame.

SAM robot mask → render-and-compare → SAM base-box mask → re-registration.
Copies inputs into --out and preserves their original files.
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np

from ..config import ROOT, SETTINGS, worker_python, runtime_env
from .common import CONFIG


def registration_python():
    import os
    if "SHOWONCE_CALIBRATION_PYTHON" in os.environ or "calibration" in SETTINGS.get("python", {}):
        return worker_python("calibration")
    return worker_python("foundationpose")


def run(cmd, env_fpose=False):
    env = runtime_env()
    env.pop("PYTHONPATH", None)
    env["PYTHONSAFEPATH"] = "1"
    ptxas = pathlib.Path("/usr/local/cuda/bin/ptxas")
    if ptxas.is_file():
        env.setdefault("TRITON_PTXAS_BLACKWELL_PATH", str(ptxas))
    env["PATH"] = str(pathlib.Path(cmd[0]).parent) + ":" + env.get("PATH", "")
    r = subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("Calibration worker failed: " + r.stdout[-2000:] + r.stderr[-4000:])
    return r.stdout


def previous_robot_box(margin=80):
    """직전 신뢰 통과 캘리브레이션(runs/snap-*)의 로봇 마스크 bbox ± margin. 화면에 다른 로봇(Go2)이 있어도 Piper를 고르게.
    카메라가 크게 옮겨졌으면 맞지 않을 수 있다 → 그때는 --robot-box로 직접 준다"""
    import cv2
    for d in sorted((ROOT / "runs").glob("snap-*"), key=lambda p: p.stat().st_mtime, reverse=True):
        f, m = d / "T_base_cam_robot.json", d / "masks/robot.png"
        if f.exists() and m.exists() and json.load(open(f)).get("trusted"):
            ys, xs = np.nonzero(cv2.imread(str(m), cv2.IMREAD_GRAYSCALE))
            if len(xs):
                print(f"  기준 캘리브레이션: {d.name}")
                return [xs.min() - margin, ys.min() - margin, xs.max() + margin, ys.max() + margin]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mount", type=pathlib.Path, default=CONFIG / "robot_mount.json")
    ap.add_argument("--src", required=True, help="캡처 폴더(color/, depth/, cam_K.txt, meta.json)")
    ap.add_argument("--feedback", default=None, help="robot_feedback.json (기본: src 또는 그 상위 폴더에서 찾음)")
    ap.add_argument("--q-deg", type=float, nargs=6, default=None)
    ap.add_argument("--grip-m", type=float, default=None, help="손가락 한쪽 이동량(m) = 전체 폭 / 2")
    ap.add_argument("--robot-box", type=float, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"),
                    help="'robot arm' 후보 중 이 영역(px)과 가장 많이 겹치는 것을 Piper로 쓴다. 기본: 직전 신뢰 통과 캘리브레이션의 로봇 마스크 영역 ± 80 px")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="New result directory")
    args = ap.parse_args()
    P = worker_python("geometry")
    F = registration_python()
    src = pathlib.Path(args.src).resolve()
    fb_path = pathlib.Path(args.feedback) if args.feedback else next(
        (p for p in [src / "robot_feedback.json", src.parent / "robot_feedback.json"] if p.exists()), None)
    q, grip, provenance = args.q_deg, args.grip_m, {}
    if fb_path is not None:
        fb = json.load(open(fb_path))
        if q is None:
            q = np.degrees(fb["q_rad"]).round(4).tolist()
        if grip is None and fb.get("gripper_full_width_m") is not None:
            grip = fb["gripper_full_width_m"] / 2
        provenance = {"feedback": str(fb_path), "feedback_source": fb.get("source"),
                      "telemetry_fresh_at_frame": fb.get("telemetry_fresh_at_frame")}
    if q is None:
        raise SystemExit("관절각이 없다: robot_feedback.json 또는 --q-deg가 필요하다")

    work = args.out
    work.mkdir(parents=True)
    for name in ["color", "depth"]:
        shutil.copytree(src / name, work / name)
    for name in ["cam_K.txt", "meta.json"]:
        shutil.copy2(src / name, work / name)
    (work / "source.json").write_text(json.dumps({"src": str(src), "q_deg": q, "grip_m": grip, **provenance}, ensure_ascii=False, indent=1))

    out = work / "T_base_cam_robot.json"
    mount = json.load(open(args.mount))
    reg = [F, "-m", "showonce_studio.calibration.register", "--run", work, "--q-deg", *q, "--out", out,
           "--table-height", mount["table_z_in_base_m"]] + (["--grip-m", grip] if grip is not None else [])
    box = args.robot_box or previous_robot_box()
    prefer = ["--prefer", "robot", *[int(v) for v in box]] if box else []
    if box:
        print(f"  Piper 후보 선택 영역(px) {[int(v) for v in box]}")
    run([P, "-m", "showonce_studio.calibration.segment", "--run", work, "--objects", CONFIG / "objects_robot.json", *prefer])
    (work / "masks/robot_base.png").unlink(missing_ok=True)
    run(reg, env_fpose=True)
    box = json.load(open(out))["base_box_px"]
    run([P, "-m", "showonce_studio.calibration.segment", "--run", work, "--box", "robot_base", *box])
    log = run(reg, env_fpose=True)
    print("\n".join(line for line in log.splitlines() if "IoU" in line or "카메라 위치" in line))
    tbl = table_check(work, out)
    cal = json.load(open(out))
    cal["table_check"] = tbl
    out.write_text(json.dumps(cal, ensure_ascii=False, indent=1))
    print(f"  테이블(로봇 밖 depth 평면): 베이스 기준 높이 {tbl['height_m'] * 1000:+.1f} mm, 베이스 z와 기울기 {tbl['tilt_deg']:.2f}°, "
          f"평면 잔차 {tbl['plane_rms_mm']:.1f} mm, 점 {tbl['n_points']}")
    print(work)


def table_check(work, calib_path):
    """로봇 밖 depth에서 가장 큰 수평 평면(테이블)을 찾아 베이스 좌표로 높이·기울기를 잰다.
    로봇을 테이블 위에서 옮기지 않으면 높이는 상수(받침대 두께)라 등록과 독립적인 검사가 된다."""
    import cv2
    from .common import backproject, load_frame, transform

    _, depth, K, _ = load_frame(work)
    robot = cv2.imread(str(work / "masks/robot_used.png"), 0) > 0
    keep = ~cv2.dilate(robot.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool) & (depth > 0.1) & (depth < 1.5)
    T = np.array(json.load(open(calib_path))["T_base_cam"])
    pts = transform(T, backproject(depth, K, keep))
    pts = pts[np.linalg.norm(pts[:, :2], axis=1) < 0.6]  # 작업 영역 근처 테이블
    rng = np.random.default_rng(0)
    P = pts[rng.choice(len(pts), min(len(pts), 40000), replace=False)]
    best, best_n = None, 0
    for _ in range(400):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n[2]) < np.cos(np.radians(15)):  # 베이스 z와 15° 안의 평면만(벽·물체 옆면 제외)
            continue
        inl = np.abs((P - a) @ n) < 0.005
        if inl.sum() > best_n:
            best, best_n = inl, int(inl.sum())
    Q = P[best]
    c = Q.mean(0)
    n = np.linalg.svd(Q - c)[2][2]
    n = n if n[2] > 0 else -n
    rms = float(np.sqrt(np.mean(((Q - c) @ n) ** 2)))
    z0 = c[2] + (n[0] * c[0] + n[1] * c[1]) / n[2]  # 평면 n·(p − c) = 0의 x = y = 0에서 z
    return {"height_m": round(float(z0), 4),
            "tilt_deg": round(float(np.degrees(np.arccos(n[2]))), 2), "plane_rms_mm": round(rms * 1000, 2), "n_points": best_n,
            "note": "베이스 원점(x=y=0) 아래 테이블 평면 높이. 로봇을 옮기지 않으면 일정해야 한다(받침대 두께)"}


if __name__ == "__main__":
    main()
