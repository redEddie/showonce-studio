"""Existing markerless render-and-compare solver with fixed encoder joints.

Run with python -m showonce_studio.calibration.register in the native CUDA worker.
"""
import argparse
import datetime
import pathlib
import sys

import cv2
import numpy as np
import torch

from .common import CALIB, backproject, inv_T, latest_run, load_frame, rpy_deg, save_json  # noqa: E402
from .robot_model import FINGER_MAX, PiperModel, project, render  # noqa: E402

# 관절 자세 후보(deg, SDK/URDF 기준). 전원이 꺼진 Piper는 접힌 자세로 내려앉는 경우가 많다
Q_PRESETS_DEG = [
    [0, 0, 0, 0, 0, 0],          # 영점(접힘)
    [0, 30, -30, 0, 0, 0],
    [0, 90, -90, 0, 0, 0],       # 팀 home 후보
    [0, 60, -120, 0, 60, 0],
]


def so3_exp(w):
    from pytorch3d.transforms import so3_exp_map
    return so3_exp_map(w)


def initial_rotations(up, yaws):
    """베이스 +z = up(카메라 좌표), yaw는 카메라 정면 방향에서 up 축으로 돌린 각."""
    f = np.array([0, 0, 1.0]) - up * up[2]
    x0 = f / np.linalg.norm(f)
    y0 = np.cross(up, x0)
    Rs = []
    for a in yaws:
        bx = np.cos(a) * x0 + np.sin(a) * y0
        Rs.append(np.stack([bx, np.cross(up, bx), up], axis=1))
    return np.stack(Rs)


def optimize(model, glctx, R0, t0, q0, g0, mask, depth, K, H, W, up, steps, obs_pts, lr_scale=1.0, vstride=4, fix=None, band=3,
             edge_dt=None, w_edge=0.5, edge_tau=8.0, table_pts=None, table_z=None, w_table=1.0):
    """obs_pts: 로봇 마스크 안 관측 3D 점(카메라 좌표, N×3). 관측점→모델 정점 최근접 거리(ICP와 같은 원리)로 depth를 강하게 맞춘다.
    band: 마스크 경계 ±band px는 실루엣 손실에서 뺀다(SAM 경계 오차 2 px가 회전을 1~3° 끌고 가는 것을 막는다).
    edge_dt: RGB 엣지까지 거리 변환(px, H×W). 주면 모델의 보이는 모서리 점이 사진 엣지 위에 오도록 당긴다.
      윤곽 안쪽 선(링크 이음새, 베이스 커넥터)까지 맞추므로 실루엣·depth만으로 풀리지 않는 베이스 yaw–joint1 상쇄를 깬다.
    table_pts/table_z: 테이블 평면 점(카메라 좌표)과 실측 받침대로 아는 베이스 기준 테이블 높이(m). 점들의 베이스 z가 table_z가
      되도록 당긴다(카메라 높이·기울기 3자유도). 테이블(나무)은 검은 로봇 표면보다 depth가 믿을 만하다."""
    from pytorch3d.ops import knn_points

    dev = model.device
    B = len(R0)
    R0 = torch.as_tensor(R0, dtype=torch.float32, device=dev)
    w = torch.zeros(B, 3, device=dev, requires_grad=True)
    t = torch.as_tensor(t0, dtype=torch.float32, device=dev).clone().requires_grad_(True)
    q = torch.as_tensor(q0, dtype=torch.float32, device=dev).clone().requires_grad_(True)
    g = torch.as_tensor(g0, dtype=torch.float32, device=dev).clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [w], "lr": 0.02 * lr_scale}, {"params": [t], "lr": 0.006 * lr_scale},
                            {"params": [q], "lr": 0.0 if (fix or {}).get("q") else 0.05 * lr_scale},
                            {"params": [g], "lr": 0.0 if (fix or {}).get("g") else 0.002 * lr_scale}])
    # 그룹별 학습률에 비례해 줄인다(eta_min을 공통으로 주면 학습률 0으로 고정한 관절도 움직인다)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * i / max(steps, 1))))
    M = torch.as_tensor(mask, dtype=torch.float32, device=dev)[None]
    pool = lambda x: torch.nn.functional.max_pool2d(x[:, None], 2 * band + 1, 1, band)[:, 0]  # noqa: E731
    M_out, M_in = pool(M), 1 - pool(1 - M)  # 확실히 배경이 아닌 곳 / 확실히 로봇인 곳
    D = torch.as_tensor(depth, dtype=torch.float32, device=dev)[None]
    upt = torch.as_tensor(up, dtype=torch.float32, device=dev)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=dev)
    if edge_dt is not None:
        DT = torch.as_tensor(np.minimum(edge_dt, edge_tau), dtype=torch.float32, device=dev)[None, None]
        DT = DT.expand(B, -1, -1, -1)
    P = torch.as_tensor(obs_pts, dtype=torch.float32, device=dev)[None].expand(B, -1, -1).contiguous()
    TP = torch.as_tensor(table_pts, dtype=torch.float32, device=dev) if table_pts is not None else None
    for it in range(steps + 1):
        R = R0 @ so3_exp(w)
        T = torch.eye(4, device=dev).repeat(B, 1, 1)
        T[:, :3, :3], T[:, :3, 3] = R, t
        V = model.vertices(q, g, T)
        sil, dr_ = render(glctx, V, model.faces, Kt, H, W)
        nn = knn_points(P, V[:, ::vstride].contiguous(), K=1).dists[..., 0].clamp(min=1e-12).sqrt()
        l_pt = (torch.nn.functional.huber_loss(nn, torch.zeros_like(nn), reduction="none", delta=0.01) / 0.01).mean(1)
        # IoU는 가려진 모델 부분(관측이 모델보다 2 cm 넘게 앞 = 앞에 다른 물체)을 빼고 잰다
        sil_v = sil * (1 - ((D > 0) & (D < dr_.detach() - 0.02)).float())
        inter = (sil_v * M).sum((1, 2))
        union = (sil_v + M - sil_v * M).sum((1, 2))
        iou = inter / union.clamp(min=1)
        # 마스크 밖 모델은 관측 depth가 모델보다 2 cm 이상 뒤일 때(= 정말 배경)만 벌점. SAM이 빠뜨린 부품(베이스 등)이나
        # 앞을 가린 물체(관측이 더 가까움), depth 결측은 판단하지 않는다
        bg = ((D > dr_.detach() + 0.02) & (D > 0)).float()
        l_sil = ((sil * (1 - M_out) * bg).sum((1, 2)) + ((1 - sil) * M_in).sum((1, 2))) / M_in.sum().clamp(min=1)
        both = (sil > 0.5) & (M > 0.5) & (D > 0) & (D > dr_.detach() - 0.02)
        err = torch.nn.functional.huber_loss(dr_, D.expand_as(dr_), reduction="none", delta=0.01) / 0.01
        l_dep = (err * both).sum((1, 2)) / both.sum((1, 2)).clamp(min=1)
        l_up = 1 - R[:, :, 2] @ upt
        l_edge = torch.zeros(B, device=dev)
        if edge_dt is not None:
            uv, z = project(model.crease_points(q, g, T), Kt)
            grid = torch.stack([2 * uv[..., 0] / W - 1, 2 * uv[..., 1] / H - 1], -1)[:, :, None]  # (B,E,1,2)
            with torch.no_grad():  # 가려진 모서리는 뺀다: 렌더링 depth보다 5 mm 이상 뒤면 안 보임
                zr = torch.nn.functional.grid_sample(dr_[:, None], grid, mode="nearest", align_corners=False)[:, 0, :, 0]
                vis = (grid.abs() < 1).all(-1)[..., 0] & (zr > 0) & (z < zr + 0.005)
            dist = torch.nn.functional.grid_sample(DT, grid, mode="bilinear", align_corners=False)[:, 0, :, 0]
            l_edge = (dist * vis).sum(1) / vis.sum(1).clamp(min=1) / edge_tau
        # IMU는 초기값과 약한 사전항으로만 쓴다(IMU·테이블 기울기 오차 ~1°를 데이터가 고칠 수 있게)
        l_table = torch.zeros(B, device=dev)
        if TP is not None:  # 베이스 z = R[:, :, 2] · (p − t)
            zb = ((TP[None] - t[:, None]) * R[:, None, :, 2]).sum(-1)
            l_table = (torch.nn.functional.huber_loss(zb, torch.full_like(zb, table_z), reduction="none", delta=0.005) / 0.005).mean(1)
        loss = l_sil + 0.3 * l_dep + 1.0 * l_pt + 0.3 * l_up + w_edge * l_edge + w_table * l_table
        if it == steps:
            break
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            if not (fix or {}).get("q"):  # 읽은 관절각은 URDF 범위를 살짝 넘을 수 있다(실측 joint2 -2°) → 고정이면 자르지 않는다
                q.copy_(torch.max(torch.min(q, model.upper_t), model.lower_t))
            g.clamp_(0, FINGER_MAX)
    with torch.no_grad():
        resid = torch.stack([(dr_[i] - D[0])[both[i]].abs().median() if both[i].any() else torch.tensor(1.0, device=dev)
                             for i in range(B)])
    out = {"T_cam_base": T.detach().cpu().numpy(), "q": q.detach().cpu().numpy(), "grip": g.detach().cpu().numpy(),
           "loss": loss.detach().cpu().numpy(), "iou": iou.detach().cpu().numpy(),
           "depth_resid_m": resid.cpu().numpy(), "edge": l_edge.detach().cpu().numpy(), "sil": sil.detach().cpu().numpy(),
           "table": l_table.detach().cpu().numpy()}
    return out


def rgb_yaw_scan(model, glctx, res, best, mask, depth, K, H, W, up, obs, dt, span=12, w_edge=1.0):
    """베이스 yaw와 joint1은 서로 상쇄돼 실루엣·depth 손실이 ±5° 안에서 평평하다(합성 검증).
    최적해에서 베이스를 +δ, joint1을 −δ 돌린 후보를 훑어 (기존 손실 + RGB 엣지 비용)이 최소인 δ를 고른다.
    엣지 비용은 지형이 얕고 기준값이 커서(사진에 안 보이는 모서리가 많음) 경사 항으로 쓰면 엉뚱한 엣지에 끌린다 → 판정에만 쓴다."""
    T0, q0, g0 = res["T_cam_base"][best], res["q"][best], res["grip"][best]
    deltas = np.radians(np.arange(-span, span + 1, 1.0))
    R0, t0, qs = [], [], []
    for a in deltas:
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        R0.append(T0[:3, :3] @ Rz)
        t0.append(T0[:3, 3])
        q = q0.copy()
        q[0] -= a
        qs.append(q)
    ev = optimize(model, glctx, np.array(R0), np.array(t0), np.array(qs), np.full(len(deltas), g0), mask, depth, K, H, W, up,
                  0, obs, vstride=2, edge_dt=dt)
    total = ev["loss"] + w_edge * ev["edge"]
    k = int(np.argmin(total))
    flat = np.degrees(deltas[ev["loss"] < ev["loss"].min() + 0.001])
    info = {"chosen_deg": int(round(np.degrees(deltas[k]))), "base_only_deg": int(round(np.degrees(deltas[np.argmin(ev["loss"])]))),
            "flat_width_deg": int(round(flat.max() - flat.min())), "edge_cost": round(float(ev["edge"][k]), 4)}
    return ev, k, info


def table_up(depth, K, robot_mask, iters=300, tol=0.006, return_points=False):
    """로봇 밖 depth 점에서 RANSAC으로 가장 큰 '수평에 가까운' 평면을 찾아 위쪽 법선(카메라 좌표)을 돌려준다.
    후보 법선은 이미지 위쪽(-y)과 60° 안이어야 한다(벽·모니터 같은 수직면 제외). 카메라 쪽을 향하게 부호를 맞춘다."""
    keep = ~cv2.dilate(robot_mask.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    keep[: depth.shape[0] // 3] = False  # 테이블은 화면 아래쪽
    pts = backproject(depth, K, keep & (depth > 0.1) & (depth < 2.0))
    if len(pts) < 500:
        raise SystemExit("IMU도 없고 테이블 depth 점도 부족해 베이스 +z를 정할 수 없다")
    rng = np.random.default_rng(0)
    P = pts[rng.choice(len(pts), min(len(pts), 30000), replace=False)]
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if n @ a > 0:  # 카메라(원점) 쪽을 향하게
            n = -n
        if n @ np.array([0, -1.0, 0]) < 0.5:
            continue
        inl = np.abs((P - a) @ n) < tol
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    if best is None:
        raise SystemExit("수평 테이블 평면을 찾지 못했다")
    Q = P[best]
    _, _, vt = np.linalg.svd(Q - Q.mean(0))
    n = vt[2]
    if n @ Q.mean(0) > 0:
        n = -n
    print(f"테이블 평면: 점 {best_n}/{len(P)}개")
    if return_points:
        return n, Q[rng.choice(len(Q), min(len(Q), 3000), replace=False)]
    return n


def filter_base_mask(model, glctx, base_mask, depth, K, run, prev_path, tol=0.03):
    """박스 프롬프트 베이스 마스크에서 1차 결과 모델 depth와 tol 넘게 다른 픽셀을 뺀다.
    베이스 앞을 물체가 가리면 SAM 3가 그 물체를 잡는다(실측: 판다 머리) → 앞을 가린 것·배경 제거."""
    import json
    prev_path = pathlib.Path(prev_path)
    if not prev_path.exists():
        return base_mask
    prev = json.loads(prev_path.read_text())
    if prev.get("run") != str(run):
        return base_mask
    T = torch.tensor(inv_T(np.array(prev["T_base_cam"]))[None], dtype=torch.float32, device="cuda")
    q = torch.tensor(np.radians(prev["robot_q_deg"])[None], dtype=torch.float32, device="cuda")
    g = torch.tensor([prev["gripper_finger_m"]], dtype=torch.float32, device="cuda")
    H, W = depth.shape
    with torch.no_grad():
        _, dm = render(glctx, model.vertices(q, g, T), model.faces, torch.tensor(K, dtype=torch.float32, device="cuda"), H, W)
    dm = dm[0].cpu().numpy()
    ok = base_mask & (depth > 0) & (dm > 0) & (np.abs(depth - dm) < tol)
    print(f"베이스 마스크: {base_mask.sum()} px 중 1차 모델 depth와 ±{tol * 100:.0f} cm 안 {ok.sum()} px만 쓴다")
    return ok


def base_box(model, T_cam_base, K, W, H, pad=0.05):
    """base_link 정점을 투영한 박스(px, 5% 여유)."""
    v = model.verts[model.part == 0].cpu().numpy().astype(np.float64)
    p = v @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
    u, w = K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2], K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]
    x0, x1, y0, y1 = u.min(), u.max(), w.min(), w.max()
    dx, dy = pad * (x1 - x0), pad * (y1 - y0)
    return [int(np.clip(x0 - dx, 0, W - 1)), int(np.clip(y0 - dy, 0, H - 1)), int(np.clip(x1 + dx, 0, W - 1)), int(np.clip(y1 + dy, 0, H - 1))]


def edge_map(rgb, mask):
    """로봇 주변 RGB 엣지(Canny). 어두운 사진은 CLAHE로 대비를 올린다. 엣지가 너무 적으면(불 꺼짐) 쓰지 않는다."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    near = cv2.dilate(mask.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    brightness = float(np.median(gray[near]))
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    v = float(np.median(g[near]))
    e = cv2.Canny(g, max(10, 0.5 * v), max(30, 1.2 * v))
    e[~near] = 0
    density = float((e[mask] > 0).mean())
    return e, {"brightness": brightness, "edge_density": density, "usable": brightness > 30 and density > 0.01}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--mask", nargs="+", default=None,
                    help="로봇 마스크 PNG, 여러 개면 합집합 (기본: <run>/masks/robot.png [+ robot_base.png 있으면])")
    ap.add_argument("--out", default=str(CALIB / "T_base_cam.json"))
    ap.add_argument("--yaws", type=int, default=12)
    ap.add_argument("--coarse-scale", type=float, default=0.4)
    ap.add_argument("--coarse-steps", type=int, default=200)
    ap.add_argument("--fine-steps", type=int, default=300)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--q-deg", type=float, nargs=6, default=None,
                    help="관절각을 알면(전원만 켜고 SDK로 읽은 값, 구동 불필요) 고정한다. 베이스 yaw–joint1 모호성이 사라진다")
    ap.add_argument("--grip-m", type=float, default=None, help="손가락 한쪽 이동량을 알면 고정")
    ap.add_argument("--table-height", type=float, default=None,
                    help="베이스 기준 테이블 면 높이(m, 받침대 두께의 음수, 예 -0.015). 주면 테이블 depth로 카메라 높이·기울기를 잡는다")
    ap.add_argument("--rgb", choices=["auto", "on", "off"], default="off",
                    help="RGB 엣지로 베이스 yaw–joint1 상쇄를 판정(관절각 모를 때만). 합성 검증에서 일관되게 낫지 않아 기본 off — "
                         "실제 사진에서 on/off 비교용. auto: 로봇 영역이 충분히 밝고 엣지가 있으면 켠다")
    args = ap.parse_args()
    run = pathlib.Path(args.run) if args.run else latest_run()
    rgb, depth, K, meta = load_frame(run)
    paths = args.mask or [p for p in [run / "masks/robot.png", run / "masks/robot_base.png"] if p.exists()]
    masks = {pathlib.Path(p).name: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0 for p in paths}
    print("마스크:", ", ".join(masks))
    import nvdiffrast.torch as dr

    glctx = dr.RasterizeCudaContext()
    model = PiperModel()
    if "robot_base.png" in masks:
        masks["robot_base.png"] = filter_base_mask(model, glctx, masks["robot_base.png"], depth, K, run, args.out)
    mask = np.any(list(masks.values()), axis=0)
    used = run / "masks/robot_used.png"  # 실제로 맞춘 마스크(시각화용)
    cv2.imwrite(str(used), mask.astype(np.uint8) * 255)
    if not mask.any():
        raise SystemExit("로봇 마스크가 비었다")
    if meta.get("gravity_imu"):
        up = np.array(meta["gravity_imu"], float)
        up /= np.linalg.norm(up)
        up_src = "imu"
    else:
        up = table_up(depth, K, mask)
        up_src = "table_plane"
        print(f"IMU 없음 → 테이블 평면 법선을 베이스 +z로 쓴다: {np.round(up, 3).tolist()}")
    table_pts = None
    if args.table_height is not None:
        _, table_pts = table_up(depth, K, mask, return_points=True)
        print(f"테이블 높이 사전항: 베이스 기준 {args.table_height * 1000:+.1f} mm, 테이블 점 {len(table_pts)}개")
    H0, W0 = depth.shape
    torch.manual_seed(0)

    edges, rgb_info = edge_map(rgb, mask)
    use_rgb = args.rgb == "on" or (args.rgb == "auto" and rgb_info["usable"])
    dt_full = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)  # 엣지는 줄이면 끊기므로 거리를 줄인다
    print(f"RGB: 로봇 영역 밝기 {rgb_info['brightness']:.0f}/255, 엣지 밀도 {rgb_info['edge_density']:.3f} → 엣지 항 {'사용' if use_rgb else '안 씀'}")

    def scaled(s):
        H, W = int(H0 * s), int(W0 * s)
        Ks = K.copy()
        Ks[:2] *= s
        m = cv2.resize(mask.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
        d = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        dt = cv2.resize(dt_full, (W, H), interpolation=cv2.INTER_AREA) * s if use_rgb else None
        return m, d, Ks, H, W, dt

    # 로봇 중심(카메라 좌표): 마스크 depth 점의 중앙값
    pts = backproject(depth, K, cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool))
    if len(pts) < 200:
        raise SystemExit(f"로봇 마스크 안 유효 depth가 부족하다({len(pts)}점)")
    center = np.median(pts, axis=0)
    rng = np.random.default_rng(0)
    obs = pts[rng.choice(len(pts), min(4000, len(pts)), replace=False)]

    yaws = np.linspace(0, 2 * np.pi, args.yaws, endpoint=False)
    Rs = initial_rotations(up, yaws)
    R0, t0, q0 = [], [], []
    with torch.no_grad():
        for qd in ([args.q_deg] if args.q_deg else Q_PRESETS_DEG):
            qr = np.radians(qd)
            Tl = model.link_transforms(torch.tensor(qr[None], dtype=torch.float32, device="cuda"),
                                       torch.tensor([0.01], device="cuda"))[0]
            vb = (Tl[model.part, :3, :3] @ model.verts[:, :, None]).squeeze(-1) + Tl[model.part, :3, 3]
            cb = vb.mean(0).cpu().numpy()
            for R in Rs:
                R0.append(R)
                t0.append(center - R @ cb)
                q0.append(qr)
    R0, t0, q0 = np.array(R0), np.array(t0), np.array(q0)
    g0 = np.full(len(R0), args.grip_m if args.grip_m is not None else 0.01)
    fix = {"q": args.q_deg is not None, "g": args.grip_m is not None}

    m, d, Ks, H, W, dt = scaled(args.coarse_scale)
    coarse = optimize(model, glctx, R0, t0, q0, g0, m, d, Ks, H, W, up, args.coarse_steps, obs, vstride=8, fix=fix,
                      band=max(1, round(3 * args.coarse_scale)), table_pts=table_pts, table_z=args.table_height)
    order = np.argsort(coarse["loss"])
    print(f"1단계: 후보 {len(R0)}개(yaw {args.yaws} × 자세 {len(R0) // args.yaws}{', 관절각 고정' if fix['q'] else ''}) → 상위 loss {coarse['loss'][order[:5]].round(3).tolist()}")

    top = order[:args.top]
    m, d, Ks, H, W, dt = scaled(1.0)
    Tc = coarse["T_cam_base"][top]
    fine = optimize(model, glctx, Tc[:, :3, :3], Tc[:, :3, 3], coarse["q"][top], coarse["grip"][top], m, d, Ks, H, W, up,
                    args.fine_steps, obs, lr_scale=0.3, vstride=2, fix=fix, table_pts=table_pts, table_z=args.table_height)
    best = int(np.argmin(fine["loss"]))
    fine0, best0 = fine, best  # 2위 후보 비교는 훑기 전 결과로
    yaw_scan = None
    if use_rgb and not fix["q"]:
        fine, best, yaw_scan = rgb_yaw_scan(model, glctx, fine, best, m, d, Ks, H, W, up, obs, dt)
        print(f"RGB yaw 판정: 베이스 yaw +δ / joint1 −δ 훑기 → δ = {yaw_scan['chosen_deg']}° "
              f"(기존 손실만 쓰면 {yaw_scan['base_only_deg']}°, 기존 손실이 평평한 폭 {yaw_scan['flat_width_deg']}°)")
    T_cam_base = fine["T_cam_base"][best].astype(np.float64)
    T_base_cam = inv_T(T_cam_base)
    others = [i for i in range(len(top)) if i != best0]
    gap = None
    if others:
        j = min(others, key=lambda i: fine0["loss"][i])
        dpos = np.linalg.norm(fine0["T_cam_base"][j][:3, 3] - fine0["T_cam_base"][best0][:3, 3]) * 1000
        gap = {"second_loss": round(float(fine0["loss"][j]), 4), "second_pos_diff_mm": round(float(dpos), 1)}
    iou, resid = float(fine["iou"][best]), float(fine["depth_resid_m"][best])
    imu = float(np.degrees(np.arccos(np.clip(T_cam_base[:3, 2] @ up, -1, 1))))
    table_err = None
    if table_pts is not None:
        Tb = inv_T(T_cam_base)
        zb = table_pts @ Tb[2, :3] + Tb[2, 3]
        table_err = float(np.median(np.abs(zb - args.table_height)))
        # 실측 근거(2026-10-21): 받침대 15 mm 조건에서 큰 큐브 50 mm → 52.6, 그릇 높이 71 → 74 mm, 그리퍼 끝 투영 약 1 cm.
        # IoU는 SAM 3 마스크 구멍 때문에 맞는 등록도 0.57~0.83이라 하한만 둔다
        trusted = iou > 0.5 and resid < 0.008 and imu < 2.0 and table_err < 0.005
        trust_rule = "table_prior: IoU>0.5, 로봇 depth 잔차<8mm, 베이스z-테이블법선<2°, 테이블 높이 잔차<5mm"
    else:
        trusted = iou > 0.8 and resid < 0.01
        trust_rule = "no_table_prior: IoU>0.8, 로봇 depth 잔차<10mm"
    result = {
        "frame": "piper_base_link",
        "T_base_cam": np.round(T_base_cam, 6).tolist(),
        "source": "robot_markerless",
        "method": "render_and_compare(urdf+mjcf_gripper, nvdiffrast)",
        "trusted": bool(trusted),
        "camera": meta["camera"],
        "serial": meta["serial"],
        "stamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "run": str(run),
        "robot_q_deg": np.round(np.degrees(fine["q"][best]), 2).tolist(),
        "robot_q_known": fix["q"],
        "gripper_finger_m": round(float(fine["grip"][best]), 4),
        "rgb_edges": bool(use_rgb),
        "rgb_yaw_scan": yaw_scan,
        "mask_iou": round(iou, 4),
        "depth_resid_median_mm": round(resid * 1000, 2),
        "up_source": up_src,
        "table_height_prior_m": args.table_height,
        "table_height_resid_mm": None if table_err is None else round(table_err * 1000, 2),
        "trust_rule": trust_rule,
        "base_z_vs_up_deg": round(imu, 2),
        "runner_up": gap,
        "mask_files": [str(used)],
        "mask_sources": [str(p) for p in paths],
        # 베이스는 SAM 3 문구로 잘 안 잡힌다 → 이 박스로 segment.py --box robot_base ... 후 다시 실행
        "base_box_px": base_box(model, T_cam_base, K, W0, H0),
        "camera_pos_in_base_m": np.round(T_base_cam[:3, 3], 4).tolist(),
        "camera_rpy_in_base_deg": np.round(rpy_deg(T_base_cam[:3, :3]), 2).tolist(),
    }
    save_json(args.out, result)

    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    sil = (fine["sil"][best] > 0.5).astype(np.uint8)
    vis[sil > 0] = (0.5 * vis[sil > 0] + [0, 110, 0]).clip(0, 255).astype(np.uint8)
    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, cnt, -1, (0, 0, 255), 2)
    rv, _ = cv2.Rodrigues(T_cam_base[:3, :3])
    cv2.drawFrameAxes(vis, K, np.zeros(5), rv, T_cam_base[:3, 3], 0.1, 3)
    check = pathlib.Path(args.out).with_name("check.png")
    cv2.imwrite(str(check), vis)

    print(f"T_base_cam → {args.out}  (시각 확인: {check} — 초록=모델, 빨강=SAM 마스크, 축=베이스)")
    print(f"  마스크 IoU {iou:.3f}, depth 잔차 중앙값 {resid * 1000:.1f} mm, 베이스 z vs {up_src} {imu:.2f}°  → {'신뢰' if trusted else '확인 필요'}")
    print(f"  추정 관절각 {result['robot_q_deg']} deg, 손가락 {result['gripper_finger_m']} m")
    print(f"  카메라 위치(베이스) {result['camera_pos_in_base_m']} m, rpy {result['camera_rpy_in_base_deg']} deg")
    if gap:
        print(f"  2위 후보 loss {gap['second_loss']} (1위 {float(fine0['loss'][best0]):.4f}), 위치 차 {gap['second_pos_diff_mm']} mm")


if __name__ == "__main__":
    main()
