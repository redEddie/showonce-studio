#!/usr/bin/env python3
"""Launch Studio markerless calibration using the installed package."""
import os
from pathlib import Path
import sys
from show_env import PYTHON, runtime_env

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    env = runtime_env()
    env.setdefault('SHOWONCE_HOME', str(root))
    config = root/'configs/studio.local.toml'
    if config.is_file():env.setdefault('SHOWONCE_CONFIG', str(config))
    env.pop('PYTHONPATH', None)
    os.execve(str(PYTHON), [str(PYTHON), '-P', '-m', 'showonce_studio.calibration.cli', *sys.argv[1:]], env)
