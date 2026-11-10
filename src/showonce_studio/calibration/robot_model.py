"""Piper 로봇 모델(torch, 미분 가능 FK + 배치 메시).

- 팔(base_link~link5) 기구학과 메시: 제조사 URDF (pyAgxArm SDK FK와 0.09 mm 일치, pac2026 hardware_audit 기록)
- 그리퍼 몸체(link6)와 손가락(link7/8): URDF에 없어서 Menagerie MJCF(piper-sim-lab)의 메시를 URDF link6 좌표계에 붙인다.
  두 모델의 link6 좌표축 방향은 같다(위치만 최대 11 mm 다름 — MJCF가 구버전 기구학).
관절 순서와 부호는 URDF(= SDK joint_deg)와 같다.
"""
import pathlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

DATA = pathlib.Path(__file__).parent / "data/robot"
URDF = DATA / "manufacturer/piper.urdf"
MJCF = DATA / "menagerie/piper.xml"
FINGER_MAX = 0.05  # 손가락 한쪽 이동 범위 (m). MJCF range는 0.035(간격 70 mm)지만 실제 그리퍼는 96.3 mm까지 열린다(SDK·자 실측)
CREASE_DEG = 35  # 이웃 면이 이 각도 이상 꺾이면 모서리(RGB 엣지로 보일 선)
EDGE_STEP = 0.003  # 모서리 위 점 간격 (m)


def _T(xyz, rpy):
    M = np.eye(4)
    M[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    M[:3, 3] = xyz
    return M


def _mj_geom_mesh(m, g):
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
    f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.geom_quat[g])
    return v @ R.reshape(3, 3).T + m.geom_pos[g], f


def _mj_body_T(m, b):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.body_quat[b])
    M = np.eye(4)
    M[:3, :3] = R.reshape(3, 3)
    M[:3, 3] = m.body_pos[b]
    return M


class PiperModel:
    """parts: 0 base_link, 1~5 link1~5, 6 link6(그리퍼 몸체), 7/8 손가락."""

    lower = np.array([-2.6179938, 0, -2.9670597, -1.7453292, -1.2217304, -2.0943951])
    upper = np.array([2.6179938, 3.1415926, 0, 1.7453292, 1.2217304, 2.0943951])

    def __init__(self, device="cuda", urdf=URDF, mjcf=MJCF):
        self.device = device
        root = ET.parse(urdf).getroot()
        joints = {j.get("name"): j for j in root.findall("joint")}
        self.joint_origin, self.joint_axis = [], []
        for i in range(6):
            j = joints[f"joint{i + 1}"]
            o = j.find("origin")
            self.joint_origin.append(_T([float(v) for v in o.get("xyz").split()], [float(v) for v in o.get("rpy").split()]))
            self.joint_axis.append([float(v) for v in j.find("axis").get("xyz").split()])
        mesh_dir = pathlib.Path(urdf).parent / "meshes"
        verts, faces, part = [], [], []
        edge_pts, edge_part = [], []
        n = 0

        def add(v, f, pid):
            nonlocal n
            verts.append(v)
            faces.append(f + n)
            part.append(np.full(len(v), pid))
            n += len(v)
            e = self._crease_points(v, f)
            edge_pts.append(e)
            edge_part.append(np.full(len(e), pid))

        for pid, name in enumerate(["base_link", "link1", "link2", "link3", "link4", "link5"]):
            mesh = trimesh.load(mesh_dir / f"{name}.stl", force="mesh")
            add(np.asarray(mesh.vertices), np.asarray(mesh.faces), pid)
        m = mujoco.MjModel.from_xml_path(str(mjcf))
        self.finger_T, self.finger_axis = [], []
        for pid, name in [(6, "link6"), (7, "link7"), (8, "link8")]:
            b = m.body(name).id
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g] == 0:
                    add(*_mj_geom_mesh(m, g), pid)
            if pid > 6:
                self.finger_T.append(_mj_body_T(m, b))
                self.finger_axis.append(m.jnt_axis[m.body_jntadr[b]].copy())
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)  # noqa: E731
        self.verts = t(np.concatenate(verts))
        self.faces = t(np.concatenate(faces), torch.int32)
        self.part = t(np.concatenate(part), torch.long)
        self.edge_pts = t(np.concatenate(edge_pts))
        self.edge_part = t(np.concatenate(edge_part), torch.long)
        self.J_origin = t(np.stack(self.joint_origin))
        self.J_axis = t(np.array(self.joint_axis))
        self.F_T = t(np.stack(self.finger_T))
        self.F_axis = t(np.array(self.finger_axis))
        self.lower_t, self.upper_t = t(self.lower), t(self.upper)

    @staticmethod
    def _crease_points(v, f):
        """메시의 날카로운 모서리(면 사이 각 > CREASE_DEG)를 EDGE_STEP 간격 점으로 샘플링 (부품 좌표)."""
        tm = trimesh.Trimesh(v, f, process=True)
        sharp = tm.face_adjacency_angles > np.radians(CREASE_DEG)
        a, b = tm.vertices[tm.face_adjacency_edges[sharp, 0]], tm.vertices[tm.face_adjacency_edges[sharp, 1]]
        pts = []
        for p0, p1 in zip(a, b):
            k = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / EDGE_STEP)))
            pts.append(p0 + (p1 - p0) * ((np.arange(k) + 0.5) / k)[:, None])
        return np.concatenate(pts) if pts else np.zeros((0, 3))

    @staticmethod
    def _rot(axis, angle):
        """axis (3,), angle (B,) → (B,4,4) 회전."""
        k = axis / axis.norm()
        K = torch.zeros(3, 3, device=axis.device)
        K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
        s, c = torch.sin(angle)[:, None, None], torch.cos(angle)[:, None, None]
        R = torch.eye(3, device=axis.device) + s * K + (1 - c) * (K @ K)
        M = torch.eye(4, device=axis.device).repeat(len(angle), 1, 1)
        M[:, :3, :3] = R
        return M

    def link_transforms(self, q, grip):
        """q (B,6) rad, grip (B,) 손가락 한쪽 이동량 m → 부품별 T_base_part (B,9,4,4)."""
        B = q.shape[0]
        M = torch.eye(4, device=q.device).repeat(B, 1, 1)
        Ts = [M]
        for i in range(6):
            M = M @ self.J_origin[i] @ self._rot(self.J_axis[i], q[:, i])
            Ts.append(M)
        T6 = Ts[6]
        for k, sgn in enumerate([1.0, -1.0]):
            S = torch.eye(4, device=q.device).repeat(B, 1, 1)
            S[:, :3, 3] = self.F_axis[k] * (sgn * grip)[:, None]
            Ts.append(T6 @ self.F_T[k] @ S)
        return torch.stack(Ts, 1)  # 0 base, 1~6 link1~6, 7/8 손가락

    def vertices(self, q, grip, T_cam_base):
        """카메라 좌표 정점 (B,V,3)."""
        T = T_cam_base[:, None] @ self.link_transforms(q, grip)  # (B,9,4,4)
        Tv = T[:, self.part]  # (B,V,4,4)
        return (Tv[..., :3, :3] @ self.verts[None, :, :, None]).squeeze(-1) + Tv[..., :3, 3]

    def crease_points(self, q, grip, T_cam_base):
        """모서리 점의 카메라 좌표 (B,E,3)."""
        T = T_cam_base[:, None] @ self.link_transforms(q, grip)
        Tv = T[:, self.edge_part]
        return (Tv[..., :3, :3] @ self.edge_pts[None, :, :, None]).squeeze(-1) + Tv[..., :3, 3]


class PiperModelMJ(PiperModel):
    """팔까지 전부 MJCF(piper-sim-lab = Menagerie) 기구학·메시로 만든 모델. SHOW-55 시뮬 vs 실기 비교용.
    부품 번호·link_transforms·vertices는 PiperModel과 같다(MJCF body 좌표 = body_pos/quat 뒤 관절 회전)."""

    def __init__(self, device="cuda", mjcf=MJCF):
        self.device = device
        m = mujoco.MjModel.from_xml_path(str(mjcf))
        self.joint_origin, self.joint_axis = [], []
        for i in range(6):
            b = m.body(f"link{i + 1}").id
            self.joint_origin.append(_mj_body_T(m, b))
            self.joint_axis.append(m.jnt_axis[m.joint(f"joint{i + 1}").id].copy())
        verts, faces, part, edge_pts, edge_part = [], [], [], [], []
        n = 0
        for pid, name in enumerate(["base_link", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "link8"]):
            b = m.body(name).id
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g] == 0:
                    v, f = _mj_geom_mesh(m, g)
                    verts.append(v)
                    faces.append(f + n)
                    part.append(np.full(len(v), pid))
                    n += len(v)
                    e = self._crease_points(v, f)
                    edge_pts.append(e)
                    edge_part.append(np.full(len(e), pid))
        self.finger_T = [_mj_body_T(m, m.body(k).id) for k in ("link7", "link8")]
        self.finger_axis = [m.jnt_axis[m.body_jntadr[m.body(k).id]].copy() for k in ("link7", "link8")]
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)  # noqa: E731
        self.verts = t(np.concatenate(verts))
        self.faces = t(np.concatenate(faces), torch.int32)
        self.part = t(np.concatenate(part), torch.long)
        self.edge_pts = t(np.concatenate(edge_pts))
        self.edge_part = t(np.concatenate(edge_part), torch.long)
        self.J_origin = t(np.stack(self.joint_origin))
        self.J_axis = t(np.array(self.joint_axis))
        self.F_T = t(np.stack(self.finger_T))
        self.F_axis = t(np.array(self.finger_axis))
        self.lower_t, self.upper_t = t(self.lower), t(self.upper)


def project(pts_cam, K):
    """(B,N,3) → 픽셀 좌표 (B,N,2), z (B,N)."""
    z = pts_cam[..., 2].clamp(min=1e-4)
    return torch.stack([K[0, 0] * pts_cam[..., 0] / z + K[0, 2], K[1, 1] * pts_cam[..., 1] / z + K[1, 2]], -1), z


def render(glctx, verts_cam, faces, K, H, W, near=0.05, far=5.0, return_rast=False):
    """nvdiffrast로 실루엣(안티앨리어싱, 미분 가능)과 depth(m) 렌더링. verts_cam (B,V,3) OpenCV 카메라 좌표."""
    import nvdiffrast.torch as dr

    x, y, z = verts_cam.unbind(-1)
    z = z.clamp(min=1e-4)
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    # 행 0이 이미지 위쪽이 되도록 ndc_y = 2v/H - 1 (nvdiffrast는 y=-1을 0번째 행에 둔다)
    zc = (z * (far + near) - 2 * far * near) / (far - near)
    clip = torch.stack([(2 * u / W - 1) * z, (2 * v / H - 1) * z, zc, z], -1).contiguous()
    rast, _ = dr.rasterize(glctx, clip, faces, (H, W))
    cover = (rast[..., 3:] > 0).float()
    sil = dr.antialias(cover, rast, clip, faces)[..., 0]
    depth = dr.interpolate(verts_cam[..., 2:].contiguous(), rast, faces)[0][..., 0] * cover[..., 0]
    return (sil, depth, rast) if return_rast else (sil, depth)
