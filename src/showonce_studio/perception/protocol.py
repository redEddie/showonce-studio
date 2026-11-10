"""Live perception file/preview protocol extracted from the working inspector."""
import json
import hashlib
import os
from pathlib import Path
import time
import threading
import numpy as np
from .transforms import quat_wxyz

def read(path):
    return json.loads(Path(path).read_text())

def write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)

def provenance(path):
    return dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())

def overlay(frame, poses, definitions, transform, intrinsic, now=None, statuses=None):
    """Reproject known-size geometry. This is a consistency image, not ground truth."""
    import cv2
    image = frame.copy()
    inverse = np.linalg.inv(transform)
    def project(points):
        points = points @ inverse[:3, :3].T + inverse[:3, 3]
        if (points[:, 2] <= 0).any():
            return None
        pixels = points[:, :2] / points[:, 2, None]
        return np.rint(pixels @ intrinsic[:2, :2].T + intrinsic[:2, 2]).astype(np.int32)
    y = 25
    for definition, colour in zip(definitions, [(50, 220, 100), (255, 180, 60)]):
        oid = definition['id']
        pose = poses.get(oid)
        status = (statuses or {}).get(oid, 'fresh' if pose is not None else 'missing')
        text = f'{oid}: {status}'
        if pose is not None:
            T, size = np.asarray(pose['T']), definition['size']
            age = '' if now is None else f' age={(now-pose["t"])*1000:.0f}ms'
            text = f'{oid}: {status} xyz(mm)={np.round(T[:3, 3]*1000, 1).tolist()}{age}'
            if status != 'fresh':
                cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .45, (50, 100, 255), 1)
                y += 22
                continue
            if size['type'] == 'box':
                corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * size['half_extents']
                pixels = project(corners @ T[:3, :3].T + T[:3, 3])
                if pixels is not None:
                    for i in range(8):
                        for j in range(i+1, 8):
                            if bin(i ^ j).count('1') == 1:
                                cv2.line(image, tuple(pixels[i]), tuple(pixels[j]), colour, 2)
            else:
                a = np.linspace(0, 2*np.pi, 100)
                for z, radius in [(0., size['bottom_radius']), (size['height'], size['rim_radius'])]:
                    ring = np.c_[radius*np.cos(a), radius*np.sin(a), np.full_like(a, z)]
                    pixels = project(ring @ T[:3, :3].T + T[:3, 3])
                    if pixels is not None:
                        cv2.polylines(image, [pixels], True, colour, 2)
            origin = project(T[:3, 3][None])
            if origin is not None:
                cv2.circle(image, tuple(origin[0]), 5, colour, -1)
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .45, colour, 1)
        y += 22
    return image

class PreviewWriter:
    """Encode only the newest frame off the camera publishing loop."""
    def __init__(self, output):
        self.output = Path(output)
        self.condition = threading.Condition()
        self.latest = None
        self.stopping = False
        self.thread = threading.Thread(target=self.run, name='web-preview', daemon=True)
        self.thread.start()

    def submit(self, frame):
        with self.condition:
            self.latest = frame
            self.condition.notify()

    def run(self):
        import cv2
        target = self.output/'latest.jpg'
        temporary = self.output/'latest.jpg.tmp'
        while True:
            with self.condition:
                while self.latest is None and not self.stopping:
                    self.condition.wait()
                if self.stopping:
                    return
                frame, self.latest = self.latest, None
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if ok:
                try:
                    temporary.write_bytes(encoded.tobytes())
                    os.replace(temporary, target)
                except OSError as error:
                    print(f'Preview write failed: {error}', file=sys.stderr, flush=True)

    def close(self):
        with self.condition:
            self.stopping = True
            self.latest = None
            self.condition.notify()
        self.thread.join(timeout=2.)

def objects_document(poses, definitions, calib):
    objects = []
    for definition in definitions:
        pose = poses.get(definition['id'])
        if pose is None:
            continue
        T = np.asarray(pose['T'])
        objects.append(dict(id=definition['id'], pos=T[:3, 3].tolist(),
                            quat_wxyz=quat_wxyz(T[:3, :3]).tolist(), size=definition['size'],
                            checks=pose['checks'], table_z_m=pose['table_z_m'],
                            estimator=pose.get('estimator', 'depth_geometry'),
                            source_frame=pose['frame'], source_time_s=pose['t']))
    return dict(frame='piper_base_link', units='m', camera_to_base=calib, objects=objects)

def save_sam_debug(output, camera, sam, definitions, calibration, result):
    """Keep the latest actual detector input and matched masks for offline diagnosis."""
    import cv2
    frame = camera.get(result['frame'])
    if frame is None:
        return
    debug = output/'sam_latest'
    for name in ('color', 'depth', 'masks'):
        (debug/name).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug/'color/0.png'), frame['bgr'])
    cv2.imwrite(str(debug/'depth/0.png'), np.clip(np.rint(frame['depth']*1000), 0, 65535).astype(np.uint16))
    np.savetxt(debug/'cam_K.txt', camera.K)
    write(debug/'meta.json', dict(serial=calibration['serial'], source_frame=frame['id'], source_time_s=frame['t']))
    write(debug/'T_base_cam_robot.json', calibration)
    segmentation = {}
    with sam.lock:
        anchors = dict(sam.anchors)
    for definition in definitions:
        oid = definition['id']
        anchor = anchors.get(oid)
        found = oid in result['detected'] and anchor is not None and anchor[0] == frame['id']
        segmentation[oid] = dict(found=found, score=result['detected'].get(oid))
        if found:
            cv2.imwrite(str(debug/'masks'/f'{oid}.png'), anchor[1].astype(np.uint8)*255)
    write(debug/'masks/segment.json', segmentation)
    write(debug/'sam_result.json', result)
