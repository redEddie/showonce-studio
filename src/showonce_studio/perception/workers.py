"""Working retained-anchor and parallel image estimators, packaged without external source paths."""
from collections import deque
import threading
import time

import cv2
import numpy as np

from .sam import Sam
from .tracking import Tracker
from piper_policy_runtime.frame_retention import AnchorFrames
from .bowl_image import BowlRejected, estimate_bowl, track_bowl_mask
from .cube_image import CubeRejected, estimate_cube, green_mask


class RetainedSam(Sam):
    """Same worker protocol as Sam; retain the precise request RGB-D frame."""
    def __init__(self, *args, **kwargs):
        self.completed = 0
        self.last_result = None
        self.inflight_since = None
        self.events = deque(maxlen=200)
        super().__init__(*args, **kwargs)

    def run(self):
        try:
            first = self.conn.recv()
            if first != 'ready':
                return self.fail(first[1] if isinstance(first, tuple) else str(first))
            self.ready.set()
            last = -1
            while not self.stop:
                frame = self.cam.wait_newer(last, timeout=.2)
                if frame is None or frame['id'] == last:
                    continue
                last = frame['id']
                self.cam.anchor_frames.retain('sam_inflight', frame)
                started = time.monotonic()
                self.inflight_since = started
                self.conn.send((frame['id'], cv2.cvtColor(frame['bgr'], cv2.COLOR_BGR2RGB)))
                frame_id, results = self.conn.recv()
                if frame_id == 'error':
                    return self.fail(results)
                if frame_id != frame['id']:
                    return self.fail('SAM returned a different source frame ID')
                duration = time.monotonic()-started
                height, width = frame['bgr'].shape[:2]
                with self.lock:
                    self.cam.anchor_frames.retain('sam_latest_result', frame)
                    for oid, (bits, score) in results.items():
                        # Publish ownership BEFORE the anchor is visible to Tracker.
                        self.cam.anchor_frames.retain(oid, frame)
                        mask = np.unpackbits(bits)[:height*width].reshape(height, width).astype(bool)
                        self.anchors[oid] = (frame_id, mask, score)
                    self.completed += 1
                    self.last_result = dict(frame=frame_id, inference_ms=round(duration*1000, 1),
                                            detected={oid:float(value[1]) for oid, value in results.items()})
                    self.events.append(dict(self.last_result))
                    self.times.append(duration)
                self.inflight_since = None
                self.cam.anchor_frames.release('sam_inflight')
                # Small interruptible pauses also allow quick, repeatable shutdown.
                until = time.monotonic()+self.period
                while not self.stop and time.monotonic() < until:
                    time.sleep(min(.05, max(0., until-time.monotonic())))
            self.conn.send(None)
        except (EOFError, BrokenPipeError, OSError) as exc:
            if not self.stop:
                self.fail(f'SAM process communication failed: {exc}')
        except Exception as exc:
            if not self.stop:
                self.fail(f'{type(exc).__name__}: {exc}')
        finally:
            self.inflight_since = None
            self.cam.anchor_frames.release('sam_inflight')


class ImageBowlTracker(Tracker):
    """Image cube/plane geometry and measured RGB motion for the bowl."""
    def __init__(self, camera, sam, objects, transform, table_hint_z):
        # Reuse receiver/thread lifecycle, not the legacy depth cube estimator.
        super().__init__(camera, sam, [], transform)
        self.cube_definition = next((o for o in objects if o['id'] == 'green_block'), None)
        self.cube_anchor_id = self.cube_anchor_time = self.cube_mask = self.cube_initial = None
        self.cube_image = dict(accepted=False, reason='waiting_for_cube_mask')
        self.bowl_definition = next(o for o in objects if o['id'] == 'bowl')
        self.table_hint_z = table_hint_z
        self.bowl_frame = self.bowl_mask = self.bowl_centre = None
        self.bowl_anchor_id, self.bowl_anchor_time = None, None
        self.bowl_image = dict(accepted=False, reason='waiting_for_bowl_mask')
        self.shared_plane = None
        self.parallel = False
        self.cube_thread = None
        self.cube_error = None
        self.cube_last_frame = -1
        self.cube_failures = 0
        self.cube_borderline = deque(maxlen=3)
        self.anchor_failure = {}

    def start(self):
        # Native OpenCV/NumPy work releases the GIL. Each object has one owner;
        # a slow/rejected bowl fit cannot postpone the next cube observation.
        self.parallel = True
        if self.cube_definition is not None:
            self.cube_thread = threading.Thread(target=self.cube_loop, daemon=True)
            self.cube_thread.start()
        super().start()

    def cube_loop(self):
        last = -1
        try:
            while not self.stop:
                frame = self.cam.wait_newer(last, timeout=.2)
                if frame is not None and frame['id'] != last:
                    last = frame['id']
                    self.process_cube(frame)
        except Exception as exc:
            self.cube_error = f'{type(exc).__name__}: {exc}'

    def join(self, timeout=None):
        super().join(timeout)
        if self.cube_thread is not None:
            self.cube_thread.join(timeout)

    def process_cube(self, frame):
        if self.cube_definition is None:
            return
        started = time.perf_counter()
        try:
            with self.sam.lock:
                anchor = self.sam.anchors.get('green_block')
            if anchor is not None and anchor[0] != self.cube_anchor_id:
                self.cube_anchor_id = anchor[0]
                source = self.cam.get(anchor[0])
                if source is None:
                    self.anchor_rejected['green_block'] += 1
                else:
                    try:
                        # Identify the source frame, then associate with current
                        # green pixels. No old mask is labelled as a new image.
                        seed = green_mask(source, anchor[1])
                        self.cube_mask = seed
                        self.cube_anchor_time = source['t']
                        self.hue['green_block'] = float(np.median(source['hsv'][seed, 0]))
                        self.reanchors['green_block'] += 1
                    except CubeRejected:
                        self.anchor_rejected['green_block'] += 1
            if self.cube_mask is None:
                raise CubeRejected('waiting_for_cube_mask')
            if frame['t']-self.cube_anchor_time > 4. or frame['t'] < self.cube_anchor_time:
                raise CubeRejected('cube_reidentification_expired')
            mask = green_mask(frame, self.cube_mask, self.hue.get('green_block'))
            # A sparse local plane may fit the bowl, yet a small tilt becomes
            # centimetre-scale height/XY error when extrapolated to the cube.
            plane = self.shared_plane
            if plane is not None and plane.get('sparse'):
                plane = None
            normal = np.array([0., 0., 1.]) if plane is None else plane['normal']
            offset = self.table_hint_z if plane is None else plane['offset']
            pose, table_z, checks = estimate_cube(self.cube_definition, mask, self.cam.K, self.T,
                                                table_normal=normal, table_offset=offset,
                                                rgb=frame['bgr'], initial=self.cube_initial)
            self.cube_initial = pose[:3, 3].copy(), checks['yaw_rad']
            if checks.get('requires_temporal_confirmation'):
                if not self.cube_borderline or self.cube_borderline[-1][0] != frame['id']:
                    self.cube_borderline.append((frame['id'], frame['t'], pose[:3, 3].copy()))
                positions = np.array([p[2] for p in self.cube_borderline])
                confirmed = (len(self.cube_borderline) == 3
                    and self.cube_borderline[-1][1]-self.cube_borderline[0][1] >= .05
                    and np.max(np.linalg.norm(positions-np.median(positions, axis=0), axis=1)) <= .003)
                if not confirmed:
                    raise CubeRejected('cube_borderline_geometry_confirming', checks)
                checks['borderline_temporally_confirmed'] = True
            else:
                self.cube_borderline.clear()
            self.cube_mask = mask
            self.cube_initial = pose[:3, 3].copy(), checks['yaw_rad']
            self.cube_failures = 0
            self.cube_image = dict(accepted=True, reason='fresh', checks=checks,
                                  estimator_ms=round((time.perf_counter()-started)*1000, 2))
            self.masks['green_block'], self.z_table['green_block'] = mask, table_z
            with self.updated:
                self.poses['green_block'] = dict(T=pose, t=frame['t'], frame=frame['id'], checks=checks,
                    estimator='RGB_known_cube_support_plane', t_arrive=frame['t_arrive'],
                    t_ready=frame['t_ready'], t_done=time.time())
                self.seen['green_block'].append((frame['t'], pose[:3, 3].copy(), np.degrees(checks['yaw_rad'])))
                self.updated.notify_all()
        except CubeRejected as exc:
            self.implausible['green_block'] += 1
            self.cube_image = dict(accepted=False, reason=exc.reason, checks=exc.checks,
                                  estimator_ms=round((time.perf_counter()-started)*1000, 2))
            self.cube_failures += 1
            if exc.reason != 'cube_borderline_geometry_confirming':
                self.cube_borderline.clear()
            # A rejected candidate may initialize an optimizer; it is never
            # published as a measured pose. Periodically search all yaw starts.
            if 'fit_centre_base_m' in exc.checks:
                self.cube_initial = np.array(exc.checks['fit_centre_base_m']), exc.checks['yaw_rad']
            if self.cube_failures % 10 == 0:
                self.cube_initial = None
        finally:
            self.cube_last_frame = frame['id']

    def estimate_image_bowl(self, frame, mask, initial=None):
        return estimate_bowl(self.bowl_definition, mask, frame['depth'], self.cam.K, self.T,
                             table_hint_z=self.table_hint_z, rgb=frame['bgr'],
                             exclusions=self.masks.get('green_block'), initial_centre=initial,
                             support=self.shared_plane)

    def process(self, frame):
        if frame is None or frame['id'] == self.last_frame:
            return
        started = time.perf_counter()
        try:
            with self.sam.lock:
                anchor = self.sam.anchors.get('bowl')
            seeded_current = False
            if anchor is not None and anchor[0] != self.bowl_anchor_id:
                self.bowl_anchor_id = anchor[0]
                source = self.cam.get(anchor[0])
                if source is None:
                    self.anchor_rejected['bowl'] += 1
                else:
                    try:
                        seed = self.estimate_image_bowl(source, anchor[1])
                        mask = (anchor[1] if source['id'] == frame['id']
                                else track_bowl_mask(source, frame, anchor[1]))
                        # Old masks are checked against current RGB edges and depth.
                        self.bowl_frame, self.bowl_mask = frame, mask
                        self.bowl_centre = seed[0][:3, 3]
                        self.bowl_anchor_time = source['t']
                        self.reanchors['bowl'] += 1
                        seeded_current = True
                    except BowlRejected as exc:
                        self.anchor_rejected['bowl'] += 1
                        self.anchor_failure['bowl'] = dict(reason=exc.reason, checks=exc.checks,
                                                           frame=anchor[0])
                        self.bowl_image = dict(accepted=False, reason=exc.reason, checks=exc.checks)
            if self.bowl_frame is None:
                return
            if self.bowl_anchor_time is None or frame['t']-self.bowl_anchor_time > 4.:
                raise BowlRejected('bowl_reidentification_expired')
            if not seeded_current and self.bowl_frame['id'] != frame['id']:
                self.bowl_mask = track_bowl_mask(self.bowl_frame, frame, self.bowl_mask)
                self.bowl_frame = frame
            T, table_z, checks = self.estimate_image_bowl(frame, self.bowl_mask, self.bowl_centre)
            if self.shared_plane is None:
                # Camera/base/table are fixed for this calibrated session.
                # Keep their measured plane even when a bowl frame is lost;
                # fresh RGB shape/edge/occlusion checks still run every frame.
                self.shared_plane = dict(normal=np.array(checks['table_normal_base']),
                    offset=checks['table_plane_offset_m'], source_time_s=frame['t'],
                    sparse=bool(checks.get('table_plane_sparse_validated', False)),
                    checks={k:v for k,v in checks.items() if k.startswith('table_')})
            self.bowl_centre = T[:3, 3]
            self.bowl_image = dict(accepted=True, reason='fresh', checks=checks,
                                   estimator_ms=round((time.perf_counter()-started)*1000, 2))
            self.masks['bowl'], self.z_table['bowl'] = self.bowl_mask, table_z
            with self.updated:
                self.poses['bowl'] = dict(T=T, t=frame['t'], frame=frame['id'], checks=checks,
                    estimator='image_silhouette_table_plane', t_arrive=frame['t_arrive'],
                    t_ready=frame['t_ready'], t_done=time.time())
                self.updated.notify_all()
        except BowlRejected as exc:
            self.implausible['bowl'] += 1
            self.bowl_image = dict(accepted=False, reason=exc.reason, checks=exc.checks,
                                  estimator_ms=round((time.perf_counter()-started)*1000, 2))
        finally:
            if not self.parallel:
                self.process_cube(frame)
            # Publish frame completion after both estimators, not at their start.
            with self.updated:
                self.last_frame, self.last_frame_t = frame['id'], frame['t']
                self.updated.notify_all()


def diagnostics(camera, sam, tracker):
    with sam.lock:
        completed = sam.completed
        last = dict(sam.last_result) if sam.last_result is not None else None
        times = list(sam.times)
    with tracker.lock:
        measured = list(tracker.poses)
    inflight = sam.inflight_since
    return dict(sam_completed=completed, sam_last_result=last,
                sam_inference_p95_ms=float(np.percentile(times, 95)*1000) if times else None,
                sam_inflight_s=round(time.monotonic()-inflight, 2) if inflight is not None else None,
                anchor_frames_retained=camera.anchor_frames.count(),
                anchor_source_frames_missing=camera.missing_anchor_reads,
                tracker_reanchors=dict(tracker.reanchors),
                tracker_anchor_rejections=dict(tracker.anchor_rejected),
                tracker_geometry_rejections=dict(tracker.implausible),
                tracker_tracking_lost=dict(tracker.lost), measured_ids=measured,
                anchor_failure=getattr(tracker, 'anchor_failure', {}),
                object_workers_parallel=getattr(tracker, 'parallel', False),
                cube_worker_frame=getattr(tracker, 'cube_last_frame', None),
                bowl_worker_frame=tracker.last_frame,
                cube_worker_error=getattr(tracker, 'cube_error', None),
                bowl_image=getattr(tracker, 'bowl_image', None),
                cube_image=getattr(tracker, 'cube_image', None))


def object_status(oid, pose, now, max_age_s, state):
    if pose is not None and 0 <= now-pose['t'] <= max_age_s:
        return 'fresh'
    if oid == 'bowl' and state.get('bowl_image'):
        return state['bowl_image']['reason']
    if oid == 'green_block' and state.get('cube_image'):
        return state['cube_image']['reason']
    last = state.get('sam_last_result')
    if last is None:
        return 'waiting_for_SAM'
    if oid not in last['detected']:
        return 'SAM_no_detection'
    if not state['tracker_reanchors'].get(oid):
        return ('anchor_depth_or_size_rejected' if state['tracker_anchor_rejections'].get(oid)
                else 'waiting_for_depth_estimate')
    if pose is not None:
        return 'stale_measurement'
    return 'tracking_or_geometry_rejected'
