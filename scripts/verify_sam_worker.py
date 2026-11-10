"""Verify actual SAM image/video inference on a private recording without opening devices."""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--rgb-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=4)
    parser.add_argument('--expected-sam-version', default='0.1.0')
    parser.add_argument('--expected-numpy-major', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if not 2 <= args.frames <= 16:
        parser.error('Use 2..16 frames for the bounded verification.')
    sources = sorted(args.rgb_dir.glob('*.png'))[:args.frames]
    if len(sources) != args.frames or not args.checkpoint.is_file():
        parser.error('Approved checkpoint and enough recorded RGB PNGs are required.')
    os.environ['SHOWONCE_SAM_CHECKPOINT'] = str(args.checkpoint.resolve())
    # Same CUDA library policy used by Studio workers; no environment installation changes.
    from showonce_studio.config import runtime_env
    clean_env = runtime_env()
    if os.environ.get('LD_LIBRARY_PATH', '') != clean_env['LD_LIBRARY_PATH']:
        # glibc caches loader search paths at process startup; changing environ in-place is insufficient.
        os.execve(os.sys.executable, [os.sys.executable, str(Path(__file__).resolve()), *os.sys.argv[1:]], clean_env)
    os.environ.setdefault('TRITON_PTXAS_BLACKWELL_PATH', '/usr/local/cuda/bin/ptxas')
    import numpy as np
    import torch
    from showonce_studio.showonce_geometry import GeometryEngine
    subprocess.run([os.sys.executable, '-m', 'pip', 'check'], check=True)
    if metadata.version('sam3') != args.expected_sam_version or int(np.__version__.split('.')[0]) != args.expected_numpy_major:
        raise RuntimeError('Unexpected model/NumPy environment; use the dedicated upstream SAM worker.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable; inference has not been verified.')
    torch.set_num_threads(4)
    started = time.monotonic()
    engine = GeometryEngine()
    engine.load()
    predictor = engine.predictor
    loaded = time.monotonic()
    objects = {}
    with tempfile.TemporaryDirectory(prefix='showonce-sam-verify-') as folder:
        for index, path in enumerate(sources):
            shutil.copyfile(path, Path(folder)/f'{index:06d}.png')
        for prompt in ('green cube', 'bowl'):
            session = predictor.handle_request({'type': 'start_session', 'resource_path': folder,
                                                'offload_video_to_cpu': True})['session_id']
            try:
                initial = predictor.handle_request({'type': 'add_prompt', 'session_id': session,
                                                   'frame_index': 0, 'text': prompt})['outputs']
                if not len(initial['out_obj_ids']):
                    raise RuntimeError(f'No detection for {prompt}; actual inference did not pass.')
                frames = []
                for response in predictor.handle_stream_request({'type': 'propagate_in_video',
                        'session_id': session, 'propagation_direction': 'forward', 'start_frame_index': 0}):
                    value = response['outputs']
                    masks = np.asarray(value['out_binary_masks'])
                    pixels = int(masks.sum())
                    scores = np.asarray(value['out_probs'])
                    if pixels < 20 or not np.isfinite(scores).all():
                        raise RuntimeError(f'Empty or non-finite tracking output for {prompt}')
                    frames.append({'index': int(response['frame_index']), 'objects': len(value['out_obj_ids']),
                                   'mask_pixels': pixels, 'max_score': float(scores.max()),
                                   'mask_shape': list(masks.shape),
                                   'mask_sha256': hashlib.sha256(np.ascontiguousarray(masks, dtype=np.uint8).tobytes()).hexdigest()})
                if {row['index'] for row in frames} != set(range(args.frames)):
                    raise RuntimeError('Tracker did not return all source frames.')
                objects[prompt] = frames
            finally:
                predictor.handle_request({'type': 'close_session', 'session_id': session})
    torch.cuda.synchronize()
    result = {'schema': 'showonce_sam_worker_validation_v1', 'status': 'passed',
              'gpu': torch.cuda.get_device_name(), 'packages': {name: metadata.version(name) for name in
                ('sam3', 'numpy', 'torch', 'torchvision', 'open3d', 'showonce-studio', 'piper-runtime')},
              'checkpoint_sha256': sha(args.checkpoint), 'source_rgb_sha256': [sha(p) for p in sources],
              'source_frames': len(sources), 'load_s': loaded-started, 'total_s': time.monotonic()-started,
              'tracked_objects': objects, 'pip_check': 'passed', 'camera_opened': False,
              'robot_commands_sent': False, 'weights_or_frames_bundled': False,
              'scope': 'Actual saved-RGB video prompting/propagation, not live camera, depth reconstruction or task success.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
