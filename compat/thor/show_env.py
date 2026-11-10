"""Shared launch settings for the isolated piper-show conda environment."""
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
PYTHON = Path.home()/'miniforge3/envs/piper-show/bin/python'


def runtime_env():
    env = dict(os.environ)
    # The interactive shell adds CUDA 13.2 libraries ahead of the PyTorch cu130
    # wheel, causing CUBLAS_STATUS_NOT_INITIALIZED on SAM inference.
    env['LD_LIBRARY_PATH'] = ':'.join(
        p for p in env.get('LD_LIBRARY_PATH', '').split(':')
        if p and not re.fullmatch(r'/usr/local/cuda[^/]*/lib64/?', p))
    env['PYTHONNOUSERSITE'] = '1'
    env.setdefault('CUDA_HOME', '/usr/local/cuda')
    env.setdefault('TORCH_CUDA_ARCH_LIST', '11.0')
    env.setdefault('TORCH_EXTENSIONS_DIR', str(ROOT/'.cache/torch_extensions'))
    env.setdefault('UV_CACHE_DIR', str(ROOT/'.cache/uv'))
    return env
