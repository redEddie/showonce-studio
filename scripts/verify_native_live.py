"""Bounded real-camera perception check with external team-source access denied. No robot commands."""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-python',type=Path,required=True)
    parser.add_argument('--worker-python',type=Path,required=True)
    parser.add_argument('--calib',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    processes=[]
    with tempfile.TemporaryDirectory(prefix='showonce-native-live-') as folder:
        root=Path(folder);(root/'runs').mkdir()
        calibration=root/'calibration.json';calibration.write_bytes(args.calib.read_bytes())
        config=root/'local.toml'
        config.write_text('[paths]\nhome = '+json.dumps(str(root))+'\nsam_checkpoint = '+json.dumps(str(args.checkpoint.resolve()))+'\n[live_vision]\npython = '+json.dumps(str(args.worker_python.absolute()))+'\n')
        os.environ.update(SHOWONCE_HOME=str(root),SHOWONCE_CONFIG=str(config),SHOWONCE_SAM_CHECKPOINT=str(args.checkpoint.resolve()))
        guard=root/'guard';guard.mkdir()
        (guard/'sitecustomize.py').write_text('''import os, sys
def audit(event,args):
    if event in ('open','os.listdir','os.scandir') and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=os.fsdecode(args[0])
        if '/songjisu/show33' in path or 'show100-ternary-dls-qp/scripts/show100' in path:
            raise PermissionError('External legacy source access denied: '+path)
class Deny:
    def find_spec(self,name,*args):
        if name.split('.')[0] in ('show33','live_shadow','pose_geom','perception_workers','bowl_image','cube_image'):
            raise ImportError('External legacy module denied: '+name)
sys.addaudithook(audit)
sys.meta_path.insert(0,Deny())
''')
        from showonce_studio.config import runtime_env
        from showonce_studio.live_vision_service import LiveVisionService
        env=dict(runtime_env(),SHOWONCE_HOME=str(root),SHOWONCE_CONFIG=str(config),
                 SHOWONCE_SAM_CHECKPOINT=str(args.checkpoint.resolve()),PYTHONPATH=str(guard),
                 OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        # Older Thor capture environments contain the native camera binding but no installed Studio.
        # Stage only this installed wheel's pure-Python namespace for that capture subprocess.
        # The perception subprocess below still uses its independently installed worker wheel.
        import showonce_studio
        packaged=root/'capture-package'
        shutil.copytree(Path(showonce_studio.__file__).parent,packaged/'showonce_studio')
        capture_env=dict(env,PYTHONPATH=os.pathsep.join([str(guard),str(packaged)]))
        capture_port=port();output=root/'perception'
        report={}
        with (root/'capture.log').open('w+') as capture_log,(root/'vision.log').open('w+') as vision_log:
            try:
                capture=subprocess.Popen([str(args.capture_python),'-u','-P','-m','showonce_studio.showonce_capture_node',
                    '--port',str(capture_port),'--out',str(root/'recordings'),'--calib',str(calibration)],
                    cwd=root,env=capture_env,stdout=capture_log,stderr=subprocess.STDOUT,start_new_session=True)
                processes.append(capture)
                deadline=time.monotonic()+30
                while True:
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{capture_port}/api/capture/state',timeout=2) as response:
                            camera=json.load(response)['camera']
                        if camera.get('status')=='ready':break
                        if camera.get('status')=='error':raise RuntimeError(camera.get('error','camera error'))
                    except OSError:
                        pass
                    if capture.poll() is not None or time.monotonic()>deadline:
                        capture_log.seek(0);raise RuntimeError('Capture startup failed: '+capture_log.read())
                    time.sleep(.2)
                manager=LiveVisionService(calibration,capture_port)
                real_spawn=subprocess.Popen
                # Instrument ownership/audit only; production ensure() builds the actual command/environment.
                def own_spawn(command,**kwargs):
                    assert 'PYTHONPATH' not in kwargs['env']
                    kwargs['env']['PYTHONPATH']=str(guard)  # test-only filesystem/import deny hooks
                    process=real_spawn(command,**kwargs)
                    processes.append(process)
                    return process
                subprocess.Popen=own_spawn
                try:
                    service=manager.ensure()
                finally:
                    subprocess.Popen=real_spawn
                vision=processes[-1];output=Path(service['output_dir'])
                active_since=None;deadline=time.monotonic()+150
                while vision.poll() is None:
                    startup=output/'startup.json'
                    if startup.is_file() and json.loads(startup.read_text()).get('phase')=='tracking_active':
                        if active_since is None:active_since=time.monotonic()
                        if time.monotonic()-active_since>=5:
                            vision.send_signal(signal.SIGINT);break
                    if time.monotonic()>deadline:raise TimeoutError('Native worker startup timed out')
                    time.sleep(.2)
                vision.wait(timeout=12)
                if not (output/'report.json').is_file():
                    raise RuntimeError('Native perception failed: '+(manager.home/'console.log').read_text())
                result=json.loads((output/'report.json').read_text())
                manifest=json.loads((output/'manifest.json').read_text())
                tracked=json.loads((output/'tracked_poses.json').read_text())
                diagnostics=json.loads((output/'diagnostics.json').read_text())
                if result['error'] or diagnostics['sam_completed']<1:
                    raise RuntimeError('Live perception error: '+str(result['error'])+'\n'+(manager.home/'console.log').read_text())
                assert manifest['implementation']=='studio_native_live_v1' and manifest['can_connection'] is False
                assert manifest['environment']['numpy'].startswith('1.26.')
                assert manifest['environment']['sam3']=='0.1.0'
                assert tracked['offline_replay'] is False and tracked['frame']=='piper_base_link'
                report={'status':'passed','source':'actual D435i RGB-D camera','external_source_access_denied':True,
                    'legacy_modules_denied':True,'production_service_ensure_used':True,'worker_environment':manifest['environment'],'implementation':manifest['implementation'],'pipeline_revision':manifest['pipeline_revision'],
                    'json_outputs':sorted(p.name for p in output.glob('*.json')),'sam_completed':diagnostics['sam_completed'],
                    'measured_ids':diagnostics['measured_ids'],'object_status':diagnostics['object_status'],
                    'latest_complete':result['latest_complete'],'pose_fields':sorted(tracked['objects'][0]) if tracked['objects']else [],
                    'camera_opened':True,'robot_commands_sent':False,'policy_executed':False,
                    'scope':'Live perception startup/transport/inference and file contract; no independent position-accuracy or robot-task evaluation.'}
            finally:
                for process in reversed(processes):
                    if process.poll()is None:
                        process.send_signal(signal.SIGINT)
                        try:process.wait(timeout=12)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid,signal.SIGTERM)
                            try:process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
        report['verification_processes_stopped']=all(p.poll()is not None for p in processes)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
