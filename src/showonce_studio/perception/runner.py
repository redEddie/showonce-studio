"""Native Studio live perception runner with the existing pose/preview file contract."""
import argparse
import datetime
import json
from importlib import metadata
import os
from pathlib import Path
import signal
import sys
import time
import urllib.request
import numpy as np
from piper_policy_runtime.perception_audit import PerceptionAudit, validate_calibration
from piper_policy_runtime.show100 import task_objects
from .protocol import read, write, provenance, overlay, PreviewWriter, objects_document, save_sam_debug

DEFAULT_OBJECTS = Path(__file__).with_name('data')/'objects_task.json'


def check_broker(url, calibration, timeout=20.):
    deadline = time.monotonic()+timeout
    while True:
        try:
            with urllib.request.urlopen(url.rstrip('/')+'/api/capture/state', timeout=2) as response:
                state = json.load(response)
            camera = state['camera']
            if camera.get('status') != 'ready' or time.time()-float(camera.get('last_frame_host_s', 0)) > .5:
                raise ValueError('Waiting for live RGB-D camera')
            if str(camera['serial']) != str(calibration['serial']):
                raise ValueError('Calibration/camera serial mismatch')
            return state
        except Exception:
            if time.monotonic() > deadline:
                raise
            time.sleep(.2)


def main(argv=None):
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-url', required=True)
    parser.add_argument('--calib', type=Path, required=True)
    parser.add_argument('--objects', type=Path, default=DEFAULT_OBJECTS)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=1000000000.)
    parser.add_argument('--startup-timeout', type=float, default=120.)
    parser.add_argument('--sam-period', type=float, default=0.)
    parser.add_argument('--sam-timeout', type=float, default=120.)
    parser.add_argument('--web-preview-fps', type=float, default=15.)
    parser.add_argument('--max-age', type=float, default=.2)
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--stationary', action='store_true')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--reference-tolerance-mm', type=float, default=10.)
    args = parser.parse_args(argv)
    values = [args.seconds,args.startup_timeout,args.sam_period,args.sam_timeout,args.web_preview_fps,args.max_age]
    if not np.isfinite(values).all() or min(args.seconds,args.startup_timeout,args.sam_timeout,args.max_age)<=0 or args.sam_period<0 or not 0<=args.web_preview_fps<=30:
        parser.error('Duration, timeouts and age must be positive and finite')
    if args.view and not os.environ.get('DISPLAY'):
        parser.error('--view requires a desktop DISPLAY')
    if args.checkpoint:
        os.environ['SHOWONCE_SAM_CHECKPOINT'] = str(args.checkpoint.expanduser().resolve())
    import cv2
    definitions = read(args.objects)['objects']
    ids = [obj['id'] for obj in definitions]
    if set(ids) != {'green_block','bowl'} or len(ids) != 2:
        parser.error('This pipeline expects green_block and bowl')
    definitions = sorted(definitions,key=lambda obj:obj['id']=='bowl')
    calib = read(args.calib)
    transform = validate_calibration(calib)
    check_broker(args.capture_url,calib)
    audit = PerceptionAudit(ids,max_age_s=args.max_age,stationary=args.stationary,
        reference=read(args.reference) if args.reference else None,
        reference_tolerance_m=args.reference_tolerance_mm/1000.)
    args.out.mkdir(parents=True,exist_ok=False)
    manifest = dict(mode='live_camera_only',pipeline_revision='image_cube_bowl_v4',
        implementation='studio_native_live_v1',bowl_estimator='image',cube_estimator='RGB_known_cube_support_plane',
        environment={'numpy':np.__version__,'sam3':metadata.version('sam3')},
        cpu_affinity=sorted(os.sched_getaffinity(0)),started_at=datetime.datetime.now().astimezone().isoformat(),
        calibration=provenance(args.calib),objects=provenance(args.objects),calibration_saved_trusted=bool(calib.get('trusted')),
        calibration_current_setup_verified=False,camera_source='studio_rgbd_broker',capture_url=args.capture_url,
        assumptions=['upright objects resting on the table','cube origin=center; bowl origin=outer bottom centre',
                     'reprojection is a consistency check, not an independent position reference'],
        hardware_commands_sent=False,policy_inference=False,can_connection=False)
    write(args.out/'manifest.json',manifest)
    write(args.out/'startup.json',dict(phase='imports_and_camera_init',time_s=time.time()))
    log = (args.out/'measurements.jsonl').open('w')
    latest_poses, latest_diagnostics = {}, None
    last_visual_save = 0.
    last_preview_submit = 0.
    preview = PreviewWriter(args.out) if args.web_preview_fps else None
    def record(poses, image, K, now, offline=False, diagnostic=None):
        nonlocal latest_poses, latest_diagnostics, last_visual_save, last_preview_submit
        document = objects_document(poses, definitions, calib)
        # Estimator transport keeps original measurement times, including
        # currently occluded objects. Fresh-only object_poses stays available
        # for perception diagnostics and strict direct-observation clients.
        write(args.out/'tracked_poses.json', dict(document, sample_time_s=now, offline_replay=offline))
        observations = {o['id']: dict(pos=o['pos'], frame=o['source_frame'], t=o['source_time_s']) for o in document['objects']}
        fresh = audit.add(now, observations, offline=offline)
        packet = dict(sample_time_s=now, fresh_ids=fresh, **document)
        if diagnostic is not None:
            latest_diagnostics = diagnostic
            packet['diagnostics'] = diagnostic
            write(args.out/'diagnostics.json', diagnostic)
        log.write(json.dumps(packet, allow_nan=False)+'\n'); log.flush()
        # Downstream consumers get only currently accepted poses, never old cached ones.
        current = dict(document, objects=[o for o in document['objects'] if o['id'] in fresh])
        current['complete'] = set(fresh) == set(ids)
        current['offline_replay'] = offline
        if current['complete']:
            task_objects(current)  # Check the SHOW-100 object-input contract.
        write(args.out/'object_poses.json', current)
        clock = time.monotonic()
        save_visual = offline or clock-last_visual_save >= (1. if preview else .2)
        send_preview = preview is not None and clock-last_preview_submit >= 1./args.web_preview_fps
        visual = (overlay(image, poses, definitions, transform, K, None if offline else now,
                          diagnostic.get('object_status') if diagnostic else None)
                  if args.view or save_visual or send_preview else None)
        # JSON is published every processed frame. PNG encoding is diagnostic
        # work, capped at 5Hz so it cannot throttle the observation stream.
        if save_visual:
            cv2.imwrite(str(args.out/'latest.png'), visual)
            last_visual_save = time.monotonic()
        if send_preview:
            preview.submit(visual)
            last_preview_submit = clock
        if args.view:
            cv2.imshow('SHOW-100 perception (camera only)', visual)
            if cv2.waitKey(1) & 0xff in (27, ord('q')):
                raise KeyboardInterrupt
        latest_poses = current
    error = None
    try:
        run_live(args,calib,transform,definitions,record)
    except KeyboardInterrupt:
        manifest['interrupted'] = True
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        print(error, file=sys.stderr)
    finally:
        if preview is not None:
            preview.close()
        log.close()
        if args.view:
            cv2.destroyAllWindows()
        report = dict(audit.summary(), mode=manifest['mode'], error=error,
                      latest_complete=bool(latest_poses.get('complete')), calibration_current_setup_verified=False,
                      diagnostics=latest_diagnostics,
                      note='Geometry constrains height/orientation using known dimensions; these are not independent measurements of all 6 pose coordinates.')
        write(args.out/'report.json', report)
        write(args.out/'manifest.json', manifest)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if error or not latest_poses.get('complete') else 0


def run_live(args, calib, transform, definitions, record):
    from .workers import RetainedSam, ImageBowlTracker, diagnostics, object_status
    from ..capture_subscriber import CaptureSubscriber
    camera = sam = tracker = None
    try:
        write(args.out/'startup.json', dict(phase='camera_starting', time_s=time.time()))
        camera = CaptureSubscriber(args.capture_url, calib['serial']); camera.start()
        sam = RetainedSam(camera, definitions, period=args.sam_period); sam.start()
        write(args.out/'startup.json', dict(phase='SAM_model_loading', time_s=time.time()))
        deadline = time.monotonic()+args.startup_timeout
        while not sam.ready.wait(.1):
            if not camera.is_alive() or time.monotonic() > deadline:
                raise RuntimeError('Camera/SAM startup failed or timed out')
        if sam.error:
            raise RuntimeError(sam.error)
        hint = calib.get('table_check', {}).get('height_m', calib.get('table_height_prior_m'))
        if hint is None:
            raise ValueError('Image bowl estimation requires a measured table height in calibration')
        tracker = ImageBowlTracker(camera, sam, definitions, transform, float(hint))
        tracker.start()
        write(args.out/'startup.json', dict(phase='tracking_active', time_s=time.time()))
        deadline, last_frame, last_print, last_saved_result = time.monotonic()+args.seconds, -1, 0., None
        while time.monotonic() < deadline:
            if sam.error or not all(x.is_alive() for x in (camera, sam, tracker)):
                raise RuntimeError(sam.error or 'A perception worker stopped')
            f = camera.wait_newer(last_frame, timeout=.2)
            if f is None or f['id'] == last_frame:
                continue
            # Snapshot partial/asynchronous object results without waiting for
            # either optimizer. Each pose retains its real source frame/time.
            with tracker.updated:
                poses = {oid: dict(p, table_z_m=tracker.z_table[oid]) for oid, p in tracker.poses.items()}
            now = time.time()
            state = diagnostics(camera, sam, tracker)
            if state['cube_worker_error'] is not None:
                raise RuntimeError('Cube worker failed: '+state['cube_worker_error'])
            if state['sam_inflight_s'] is not None and state['sam_inflight_s'] > args.sam_timeout:
                raise RuntimeError('SAM inference timed out; check diagnostics.json for its last completed result')
            state['object_status'] = {o['id']:object_status(o['id'], poses.get(o['id']), now, args.max_age, state) for o in definitions}
            record(poses, f['bgr'], camera.K, now, diagnostic=state)
            result = state['sam_last_result']
            if result is not None and result['frame'] != last_saved_result:
                save_sam_debug(args.out, camera, sam, definitions, calib, result)
                last_saved_result = result['frame']
            last_frame = f['id']
            if now-last_print > 1.:
                print(dict(status=state['object_status'], sam=state['sam_last_result'],
                           source_frames_missing=state['anchor_source_frames_missing'],
                           positions_mm={oid:np.round(p['T'][:3, 3]*1000, 1).tolist() for oid, p in poses.items()
                                         if state['object_status'][oid] == 'fresh'}), flush=True)
                last_print = now
    finally:
        for worker in (tracker, sam, camera):
            if worker is not None:
                worker.stop = True
        for worker in (tracker, sam, camera):
            if worker is not None:
                worker.join(timeout=2.)
        if sam is not None and sam.proc.is_alive():
            sam.proc.terminate(); sam.proc.join(timeout=2.)
            if sam.proc.is_alive():
                sam.proc.kill(); sam.proc.join(timeout=2.)
        if sam is not None:
            sam.join(timeout=1.)
            sam.conn.close()
        if camera is not None and camera.is_alive():
            try:
                camera.pipe.stop()
            except RuntimeError:
                pass
