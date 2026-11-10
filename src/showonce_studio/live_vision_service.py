"""Persistent perception worker ownership; independent of web/robot jobs."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from .config import ROOT,SETTINGS,PACKAGE_ROOT,runtime_env,sam_checkpoint,worker_python
from .showonce_stages import read_json,write_json


class LiveVisionService:
    def __init__(self,calibration,capture_port):
        config=SETTINGS.get('live_vision',{})
        self.calibration=Path(calibration).resolve();self.url=os.environ.get('SHOWONCE_CAPTURE_URL',f'http://127.0.0.1:{capture_port}')
        self.python=os.environ.get('SHOWONCE_VISION_PYTHON',config.get('python'))
        if self.python is None and SETTINGS.get('python',{}).get('geometry'):
            self.python=str(worker_python('geometry'))
        # Do not resolve executable symlinks: that would bypass the venv's pyvenv.cfg.
        if self.python is not None:self.python=str(Path(self.python).expanduser().absolute())
        self.objects=Path(os.environ.get('SHOWONCE_LIVE_OBJECTS',config.get('objects',str(PACKAGE_ROOT/'perception/data/objects_task.json')))).expanduser().resolve()
        self.checkpoint=sam_checkpoint().resolve()
        self.enabled=os.environ.get('SHOWONCE_LIVE_AUTOSTART','1'if config.get('autostart')else'0')=='1'
        self.home=ROOT/'runs/showonce-live-vision';self.home.mkdir(parents=True,exist_ok=True)

    def state(self):
        value=read_json(self.home/'service.json',{})
        alive=False
        if value.get('pid'):
            try:
                birth=Path(f'/proc/{value["pid"]}/stat').read_text().rsplit(')',1)[1].split()[19]
                command=Path(f'/proc/{value["pid"]}/cmdline').read_bytes()
                alive=birth==value.get('process_birth')and value['output_dir'].encode()in command
            except (OSError,IndexError):pass
        implementation=value.get('implementation','legacy_external_inspector' if alive else 'studio_native_live_v1')
        return dict(value,alive=alive,configured=bool(self.python),implementation=implementation,
                    external_source_required=bool(alive and implementation!='studio_native_live_v1'))

    def ensure(self):
        with (self.home/'owner.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            current=self.state()
            if current['alive']:return current
            if not current['configured']:return current
            for file in (self.python,self.calibration,self.objects,self.checkpoint):
                if not Path(file).is_file():raise ValueError('실시간 인식 구성 파일이 없습니다: '+str(file))
            folder=ROOT/'runs'/('show100-vision-studio-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
            command=[self.python,'-u','-P','-m','showonce_studio.showonce_live_vision',
                '--capture-url',self.url,'--calib',str(self.calibration),'--objects',str(self.objects),
                '--checkpoint',str(self.checkpoint),'--out',str(folder)]
            env=runtime_env()
            # Each worker has its own installed Studio/Runtime and numerical libraries.
            # Prefixing the app's site-packages would silently override the worker's NumPy.
            env.pop('PYTHONPATH',None)
            env.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                SHOW100_CPU_AFFINITY=','.join(str(i)for i in sorted(os.sched_getaffinity(0))[:-2]))
            with (self.home/'console.log').open('a')as log:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            birth=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
            write_json(self.home/'service.json',dict(pid=process.pid,process_birth=birth,output_dir=str(folder),
                calibration=str(self.calibration),capture_url=self.url,started_s=time.time(),hardware_commands_sent=False,
                implementation='studio_native_live_v1',external_source_required=False))
            return self.state()
