"""Machine-local paths and worker interpreters, independent of training checkouts."""
from pathlib import Path
import os
import re
import sys
import tomllib

PACKAGE_ROOT=Path(__file__).resolve().parent
REPOSITORY_ROOT=PACKAGE_ROOT.parents[1]
_config_path=Path(os.environ.get('SHOWONCE_CONFIG',str(REPOSITORY_ROOT/'configs/local.toml'))).expanduser()
SETTINGS=tomllib.loads(_config_path.read_text())if _config_path.is_file()else {}
paths=SETTINGS.get('paths',{})
ROOT=Path(os.environ.get('SHOWONCE_HOME',paths.get('home',str(Path.home()/'.local/share/showonce-studio')))).expanduser().resolve()
WEB_ROOT=PACKAGE_ROOT/'web'
THIRD_PARTY_ROOT=Path(os.environ.get('SHOWONCE_THIRD_PARTY',paths.get('third_party',str(ROOT/'third_party')))).expanduser().resolve()
ROOT.joinpath('runs').mkdir(parents=True,exist_ok=True)

def worker_python(kind):
    return Path(os.environ.get('SHOWONCE_'+kind.upper()+'_PYTHON',SETTINGS.get('python',{}).get(kind,sys.executable))).expanduser()
PYTHON=worker_python('capture')

def models_root():return Path(os.environ.get('SHOWONCE_MODELS',paths.get('models',str(Path.home()/'models')))).expanduser()
def sam_checkpoint():return Path(os.environ.get('SHOWONCE_SAM_CHECKPOINT',paths.get('sam_checkpoint',str(models_root()/'sam3/sam3.pt')))).expanduser()
def default_calibration():return Path(os.environ.get('SHOWONCE_CALIB',paths.get('calibration',str(ROOT/'calib/current.json')))).expanduser()
def service_port(kind,default):return int(os.environ.get('SHOWONCE_'+kind.upper()+'_PORT',SETTINGS.get('ports',{}).get(kind,default)))

def runtime_env():
    env=dict(os.environ)
    env['LD_LIBRARY_PATH']=':'.join(p for p in env.get('LD_LIBRARY_PATH','').split(':')if p and not re.fullmatch(r'/usr/local/cuda[^/]*/lib64/?',p))
    env['PYTHONNOUSERSITE']='1';env['PYTHONSAFEPATH']='1';env['SHOWONCE_HOME']=str(ROOT)
    if _config_path.is_file():env['SHOWONCE_CONFIG']=str(_config_path)
    return env
