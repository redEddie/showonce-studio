"""Compatibility entry to the standalone scripted grasp tutorial."""
from pathlib import Path
import os
import sys
workspace=Path(__file__).resolve().parent
os.environ.setdefault('SHOWONCE_HOME',str(workspace))
sys.path[:0]=[str(workspace.parent/'projects/showonce-studio/src'),str(workspace.parent/'projects/piper-runtime/src')]
from showonce_studio.scripted_grasp import main

if __name__=='__main__':main()
