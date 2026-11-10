#!/usr/bin/env python3
"""Compatibility entry point for the packaged Studio robot utilities."""
from pathlib import Path
import os,sys
from show_env import PYTHON,runtime_env
if sys.version_info[:2]!=(3,12):os.execve(str(PYTHON),[str(PYTHON),__file__,*sys.argv[1:]],runtime_env())
workspace=Path(__file__).resolve().parent
os.environ.setdefault('SHOWONCE_HOME',str(workspace))
projects=workspace.parent/'projects'
sys.path[:0]=[str(projects/'showonce-studio/src'),str(projects/'piper-runtime/src')]
from showonce_studio.showonce_policy_runtime import main
if __name__=='__main__':main()
