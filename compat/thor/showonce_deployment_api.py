"""Operating compatibility bridge to the versioned Studio package."""
from pathlib import Path
import os
import sys
from show_env import ROOT
os.environ.setdefault('SHOWONCE_HOME',str(ROOT))
paths=[str(ROOT.parent/'projects/showonce-studio/src'),str(ROOT.parent/'projects/piper-runtime/src')]
sys.path[:0]=paths
os.environ.setdefault('SHOWONCE_POLICY_PYTHON',str(ROOT/'.venv-sam3d/bin/python'))
os.environ['PYTHONPATH']=os.pathsep.join(paths+[os.environ.get('PYTHONPATH','')])
os.environ.setdefault('SHOWONCE_LIVE_AUTOSTART','1')
from showonce_studio.config import worker_python as _studio_worker_python
os.environ.setdefault('SHOWONCE_VISION_PYTHON',str(_studio_worker_python('geometry')))
from showonce_studio.showonce_deployment_api import *
