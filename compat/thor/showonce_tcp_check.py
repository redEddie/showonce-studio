"""Compatibility entry to the passive ShowOnce TCP inspection tool."""
import os
from pathlib import Path
import sys
root=Path(__file__).resolve().parent
os.environ.setdefault('SHOWONCE_HOME',str(root))
sys.path[:0]=[str(root.parent/'projects/showonce-studio/src'),str(root.parent/'projects/piper-runtime/src')]
from showonce_studio.tcp_diagnostic import main

if __name__=='__main__':main()
