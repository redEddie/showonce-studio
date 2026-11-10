from showonce_studio.config import ROOT
from showonce_studio.config import default_calibration
#!/usr/bin/env python3
"""Single-owner D435i capture node. No SAM, policy, CAN or robot commands."""
import argparse
import datetime as dt
import fcntl
import hashlib
import io
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import shutil
import threading
import time
import uuid
from urllib.parse import parse_qs,urlsplit

import cv2
import numpy as np


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


class Capture:
    def __init__(self, root, calib, serial=None):
        self.root, self.calib, self.serial = Path(root), Path(calib), serial
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.quit = threading.Event()
        self.preview = None
        self.rgbd_frame=None
        self.rgbd_sequence=0
        self.rgbd_stream_id=uuid.uuid4().hex
        self.rgbd_updated=threading.Condition(self.lock)
        self.camera = {'status': 'starting'}
        self.job = {'status': 'idle'}
        self.session = None
        self.writer = None
        self.thread = threading.Thread(target=self.camera_loop, daemon=True)

    def start(self, name='', seconds=120):
        with self.lock:
            if self.camera['status'] != 'ready':
                raise ValueError('카메라가 준비되지 않았습니다: '+str(self.camera.get('error', '시작 중')))
            if time.time()-self.camera.get('last_frame_host_s', 0) > 1.:
                raise ValueError('최신 RGB-D 프레임을 기다리고 있습니다.')
            if self.job['status'] in ('recording', 'saving'):
                raise ValueError('이미 녹화하거나 저장하고 있습니다.')
            if not np.isfinite(seconds) or not 1 <= seconds <= 600:
                raise ValueError('녹화 제한 시간은 1~600초입니다.')
            if shutil.disk_usage(self.root).free < 2*1024**3:
                raise ValueError('저장 공간이 2GB 미만입니다.')
            stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            directory = self.root/stamp
            directory.mkdir()
            (directory/'rgb').mkdir()
            (directory/'depth').mkdir()
            calibration = None
            if self.calib.is_file():
                raw = self.calib.read_bytes()
                parsed = json.loads(raw)
                if str(parsed.get('serial', '')) != self.camera['serial']:
                    raise ValueError('보정 파일과 카메라 시리얼이 다릅니다.')
                (directory/'calibration.json').write_bytes(raw)
                calibration = {'file': 'calibration.json', 'sha256': hashlib.sha256(raw).hexdigest(),
                               'camera_placement_verified_by_operator': False}
            now = time.time()
            manifest = {'schema': 'showonce_rgbd_demonstration_v1', 'id': stamp,
                        'task_name': str(name)[:200], 'demonstrator': 'human_hand',
                        'started_host_time_s': now, 'status': 'recording', 'complete': False,
                        'camera': dict(self.camera), 'calibration': calibration,
                        'rgb': {'format': 'lossless PNG', 'channel_order': 'RGB when decoded as RGB'},
                        'depth': {'format': 'lossless uint16 PNG', 'aligned_to': 'color',
                                  'meters_per_unit': self.camera['depth_scale_m'], 'invalid_value': 0,
                                  'hole_filling': False, 'temporal_filter': False},
                        'frame_index': 'frames.jsonl', 'video': None,
                        'timestamp_reference': 'per-frame RGB/depth device timestamps and host monotonic/wall times; MP4 timeline referenced to first RGB timestamp',
                        'maximum_seconds': seconds, 'robot_commands_sent': False}
            atomic_json(directory/'manifest.json', manifest)
            session = {'dir': directory, 'manifest': manifest, 'queue': queue.Queue(maxsize=90),
                       'done': threading.Event(), 'started': time.monotonic(),
                       'max_seconds': seconds, 'queued': 0, 'written': 0, 'error': None}
            self.session = session
            self.job = {'status': 'recording', 'id': stamp, 'directory': str(directory),
                        'task_name': str(name)[:200], 'frames': 0, 'elapsed_s': 0., 'max_seconds': seconds}
            self.writer = threading.Thread(target=self.write_session, args=(session,), daemon=True)
            self.writer.start()
            return dict(self.job)

    def stop(self, error=None):
        with self.lock:
            if self.session is not None:
                if error:
                    self.session['error'] = str(error)
                self.session['done'].set()
                self.job['status'] = 'saving'
                self.session = None
            return dict(self.job)

    def submit(self, bgr, depth, timestamps):
        with self.lock:
            session = self.session
            if session is None:
                return
            elapsed = time.monotonic()-session['started']
            if elapsed >= session['max_seconds']:
                self.stop()
                return
            if session['queued'] % 30 == 0 and shutil.disk_usage(self.root).free < 1024**3:
                self.stop('저장 공간이 1GB 미만이라 녹화를 종료했습니다.')
                return
            try:
                session['queue'].put_nowait((bgr.copy(), depth.copy(), dict(timestamps)))
                session['queued'] += 1
                self.job.update(elapsed_s=round(elapsed, 2), queue_size=session['queue'].qsize())
            except queue.Full:
                self.stop('저장 대기열이 가득 찼습니다. 이 녹화는 불완전 상태로 표시합니다.')

    def write_session(self, session):
        directory, rows = session['dir'], []
        try:
            with (directory/'frames.jsonl').open('w') as index:
                while not session['done'].is_set() or not session['queue'].empty():
                    try:
                        bgr, depth, row = session['queue'].get(timeout=.2)
                    except queue.Empty:
                        continue
                    number = len(rows)
                    rgb_path, depth_path = f'rgb/{number:06d}.png', f'depth/{number:06d}.png'
                    if not cv2.imwrite(str(directory/rgb_path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                        raise OSError('RGB 저장 실패')
                    if not cv2.imwrite(str(directory/depth_path), depth, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                        raise OSError('깊이 저장 실패')
                    row.update(index=number, rgb=rgb_path, depth=depth_path)
                    rows.append(row)
                    index.write(json.dumps(row, allow_nan=False)+'\n')
                    index.flush()
                    session['written'] += 1
                    with self.lock:
                        self.job['frames'] = len(rows)
            if len(rows) < 2:
                raise ValueError('저장된 프레임이 2개 미만입니다.')
            source_ms = [r['color_device_timestamp_ms'] for r in rows]
            if any(b <= a for a, b in zip(source_ms, source_ms[1:])):
                raise ValueError('RGB 장치 타임스탬프가 증가하지 않습니다.')
            # Thor's system ffmpeg lacks concat/libx264. PyAV bundles a complete
            # CPU encoder and preserves each source frame's timestamp as VFR PTS.
            import av
            time_base = Fraction(1, 1000000)
            with av.open(str(directory/'rgb.mp4'), 'w', options={'movflags': '+faststart'}) as video:
                stream = video.add_stream('libx264', rate=30)
                intr = session['manifest']['camera']['intrinsics']
                stream.width, stream.height = intr['width'], intr['height']
                stream.pix_fmt = 'yuv420p'
                stream.time_base = stream.codec_context.time_base = time_base
                stream.codec_context.thread_count = 2
                stream.options = {'preset': 'veryfast', 'crf': '18'}
                for row in rows:
                    frame = av.VideoFrame.from_ndarray(cv2.imread(str(directory/row['rgb'])), format='bgr24')
                    frame.pts = round((row['color_device_timestamp_ms']-source_ms[0])*1000)
                    frame.time_base = time_base
                    for packet in stream.encode(frame):
                        video.mux(packet)
                for packet in stream.encode():
                    video.mux(packet)
            session['manifest'].update(video='rgb.mp4', duration_s=float((source_ms[-1]-source_ms[0])/1000),
                                       average_saved_fps=float(1000/np.mean(np.diff(source_ms))),
                source_color_frames_skipped=sum(max(0, b['color_frame_number']-a['color_frame_number']-1)
                                                for a, b in zip(rows, rows[1:])))
        except Exception as error:
            session['error'] = session['error'] or str(error)
            with self.lock:
                if self.session is session:
                    self.stop(session['error'])
        finally:
            manifest = session['manifest']
            manifest.update(status='error' if session['error'] else 'complete',
                            complete=not bool(session['error']), error=session['error'],
                            frames_written=len(rows), frames_queued=session['queued'],
                            finished_host_time_s=time.time())
            atomic_json(directory/'manifest.json', manifest)
            with self.lock:
                self.job.update(status=manifest['status'], frames=len(rows), error=session['error'],
                    video_url=f"/recordings/{directory.name}/rgb.mp4" if manifest.get('video') else None)

    def camera_loop(self):
        pipeline = None
        camera_lock = None
        try:
            import pyrealsense2 as rs
            camera_lock = (ROOT/'runs/.show100-vision-camera.lock').open('a')
            try:
                fcntl.flock(camera_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('다른 인식 노드가 카메라를 사용 중입니다. 해당 노드 종료 후 촬영 앱을 다시 실행하세요.')
            pipeline = rs.pipeline()
            config = rs.config()
            if self.serial:
                config.enable_device(self.serial)
            config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
            profile = pipeline.start(config)
            device = profile.get_device()
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            with self.lock:
                self.camera = {'status': 'ready', 'serial': device.get_info(rs.camera_info.serial_number),
                    'model': device.get_info(rs.camera_info.name),
                    'depth_scale_m': device.first_depth_sensor().get_depth_scale(),
                    'intrinsics': {'width': intr.width, 'height': intr.height, 'fx': intr.fx, 'fy': intr.fy,
                                   'cx': intr.ppx, 'cy': intr.ppy, 'distortion_model': str(intr.model),
                                   'distortion_coefficients': list(intr.coeffs)}, 'nominal_fps': 30}
            align = rs.align(rs.stream.color)
            last_preview = 0.
            while not self.quit.is_set():
                frames = pipeline.wait_for_frames(timeout_ms=3000)
                host_wall, host_mono = time.time(), time.monotonic()
                aligned = align.process(frames)
                color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
                if not color or not depth:
                    continue
                bgr = np.asanyarray(color.get_data())
                z16 = np.asanyarray(depth.get_data())
                stamps = {'host_wall_time_s': host_wall, 'host_monotonic_s': host_mono,
                    'color_device_timestamp_ms': color.get_timestamp(),
                    'depth_device_timestamp_ms': depth.get_timestamp(),
                    'color_timestamp_domain': str(color.get_frame_timestamp_domain()),
                    'depth_timestamp_domain': str(depth.get_frame_timestamp_domain()),
                    'color_frame_number': color.get_frame_number(), 'depth_frame_number': depth.get_frame_number()}
                self.submit(bgr, z16, stamps)
                self.publish_rgbd(bgr,z16,stamps)
                if host_mono-last_preview >= 1/15:
                    ok, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    with self.lock:
                        if ok:
                            self.preview = jpeg.tobytes()
                        self.camera.update(last_frame_host_s=host_wall,
                                           valid_depth_fraction=float(np.count_nonzero(z16)/z16.size))
                    last_preview = host_mono
        except Exception as error:
            with self.lock:
                self.camera.update(status='error', error=str(error))
                self.stop('카메라 오류: '+str(error))
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except RuntimeError:
                    pass
            if camera_lock:
                camera_lock.close()

    def publish_rgbd(self,bgr,z16,stamps):
        # Retain an owned copy: the RealSense frame buffers can be reused.
        frame=(bgr.copy(),z16.copy(),dict(stamps))
        with self.rgbd_updated:
            self.rgbd_sequence+=1;self.rgbd_frame=frame
            self.rgbd_updated.notify_all()

    def rgbd(self,after=-1,timeout=1.):
        with self.rgbd_updated:
            self.rgbd_updated.wait_for(lambda:self.rgbd_frame is not None and self.rgbd_sequence!=after,timeout)
            if self.rgbd_frame is None:raise ValueError('RGB-D 프레임 대기 중')
            bgr,z16,stamps=self.rgbd_frame;sequence=self.rgbd_sequence;camera=dict(self.camera)
        if time.time()-stamps['host_wall_time_s']>.5:raise ValueError('RGB-D 카메라 프레임 만료')
        metadata=dict(schema='showonce_live_rgbd_v1',stream_id=self.rgbd_stream_id,sequence=sequence,
            camera=camera,stamps=stamps,aligned_to='color',offline_replay=False)
        payload=io.BytesIO()
        np.savez(payload,bgr=bgr,depth_z16=z16,metadata=np.frombuffer(json.dumps(metadata).encode(),np.uint8))
        return payload.getvalue()

    def state(self):
        with self.lock:
            recordings = []
            for directory in sorted(self.root.iterdir(), reverse=True)[:20]:
                if not directory.is_dir():
                    continue
                try:
                    manifest = json.loads((directory/'manifest.json').read_text())
                    recordings.append({k: manifest.get(k) for k in ('id', 'task_name', 'status', 'frames_written', 'duration_s', 'video', 'error')})
                except (OSError, ValueError):
                    continue
            return {'camera': dict(self.camera), 'recording': dict(self.job), 'recordings': recordings,
                    'free_gb': round(shutil.disk_usage(self.root).free/1024**3, 1), 'root': str(self.root),
                    'calibration_attached': self.calib.is_file()}

    def close(self):
        self.quit.set()
        self.thread.join(timeout=5)
        self.stop()
        if self.writer:
            self.writer.join(timeout=310)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', type=int, default=8768)
    ap.add_argument('--out', type=Path, default=ROOT/'datasets/demonstrations')
    ap.add_argument('--calib', type=Path, default=default_calibration())
    ap.add_argument('--serial')
    args = ap.parse_args()
    capture = Capture(args.out, args.calib, args.serial)
    capture.thread.start()

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, value, mime='application/json'):
            body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if urlsplit(self.path).path=='/api/capture/rgbd':
                try:
                    after=int(parse_qs(urlsplit(self.path).query).get('after',['-1'])[0])
                    return self.send(200,capture.rgbd(after),'application/x-npz')
                except (ValueError,OSError)as error:return self.send(503,{'error':str(error)})
            if self.path.split('?')[0] == '/api/capture/state':
                return self.send(200, capture.state())
            if self.path.split('?')[0] == '/capture.jpg':
                with capture.lock:
                    preview = capture.preview
                return self.send(200 if preview else 503, preview or b'Camera starting', 'image/jpeg')
            return self.send(404, {'error': 'not found'})

        def do_POST(self):
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 <= length <= 4096:
                    raise ValueError('request too large')
                body = json.loads(self.rfile.read(length) or b'{}')
                if self.path == '/api/capture/start':
                    return self.send(200, capture.start(body.get('name', ''), float(body.get('seconds', 120))))
                if self.path == '/api/capture/stop':
                    return self.send(200, capture.stop())
                return self.send(404, {'error': 'not found'})
            except (ValueError, OSError, TypeError) as error:
                return self.send(409, {'error': str(error)})

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Capture node http://127.0.0.1:{args.port} | {args.out}', flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        capture.close()


if __name__ == '__main__':
    main()
