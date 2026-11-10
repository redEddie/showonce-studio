"""Verify NumPy-2 Studio -> HTTP -> NumPy-1 SAM -> JSON/PNG files on recorded data."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-python', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    if not np.__version__.startswith('2.'):
        parser.error('Run this client in the Studio/Runtime NumPy 2 environment.')
    with tempfile.TemporaryDirectory(prefix='showonce-sam-ipc-') as folder:
        root = Path(folder)
        recordings = root/'data/datasets/demonstrations'
        recording = recordings/'20990101-000000-000002'
        (recording/'rgb').mkdir(parents=True)
        (recording/'depth').mkdir()
        rows = [json.loads(line) for line in (args.recording/'frames.jsonl').read_text().splitlines()[:4]]
        manifest = json.loads((args.recording/'manifest.json').read_text())
        manifest.update(id=recording.name, complete=True, frames_written=4)
        (recording/'manifest.json').write_text(json.dumps(manifest))
        (recording/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        for row in rows:
            for field in ('rgb', 'depth'):
                shutil.copyfile(args.recording/row[field], recording/row[field])
        config = root/'local.toml'
        config.write_text('[paths]\nhome = '+json.dumps(str(root/'data'))+'\n')
        os.environ.update(SHOWONCE_HOME=str(root/'data'), SHOWONCE_CONFIG=str(config),
                          SHOWONCE_SAM_CHECKPOINT=str(args.checkpoint.resolve()))
        from showonce_studio.config import runtime_env
        from showonce_studio.showonce_geometry_api import GeometryManager
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        manager = GeometryManager(recordings, port=port)
        with (root/'worker.log').open('w+') as log:
            process = subprocess.Popen([str(args.worker_python), '-u', '-m',
                'showonce_studio.showonce_geometry_node', '--out', str(recordings), '--port', str(port)],
                cwd=root, env=runtime_env(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic()+180
                while True:
                    if process.poll() is not None:
                        log.seek(0)
                        raise RuntimeError(log.read())
                    try:
                        health = manager.request('/health')
                        break
                    except (OSError, ValueError):
                        if time.monotonic() > deadline:
                            raise TimeoutError('SAM HTTP node startup timed out')
                        time.sleep(.1)
                if not health['environment']['numpy'].startswith('1.26.') or health['environment']['sam3'] != '0.1.0':
                    raise RuntimeError('Worker is not using unmodified SAM with NumPy 1.26')
                manager.request('/api/geometry/masks/start', {'recording_id': recording.name})
                while True:
                    state = manager.request('/api/geometry/masks/state?id='+recording.name)
                    if state['job']['status'] == 'error':
                        raise RuntimeError(state['job']['error'])
                    if state['job']['status'] == 'complete' and state['tracking_ready']:
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError('SAM mask job timed out')
                    time.sleep(.2)
                tracking = json.loads((recording/'geometry/rgbd_surface_v1/tracking.json').read_text())
                if not tracking['complete'] or len(tracking['frames']) != 4:
                    raise RuntimeError('Incomplete file protocol output')
                areas = {oid: [row['objects'][oid]['area_px'] for row in tracking['frames']]
                         for oid in ('green_block', 'bowl')}
                if any(area < 20 for values in areas.values() for area in values):
                    raise RuntimeError('Empty object masks in file output')
                after = manager.request('/health')
                result = {'status': 'passed', 'client_numpy': np.__version__,
                          'worker_environment': after['environment'], 'transport': 'existing HTTP endpoints + JSON/PNG files',
                          'object_frames': areas, 'model_retained': after['model_retained'],
                          'one_worker_process': process.poll() is None, 'camera_opened': False, 'robot_commands_sent': False}
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
        result['verification_worker_stopped'] = process.poll() is not None
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
