"""Exercise installed wheels on synthetic data; no camera/model/robot worker starts."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--deny-path', type=Path, action='append', default=[], help='Reject legacy directory access in every test subprocess')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    def python_command(*argv):
        if not args.deny_path:return [sys.executable,*argv]
        # Hooks run inside each child, before importing Studio/Runtime. Keep the
        # real sys.executable so native library resolution stays unchanged.
        bootstrap = r"""
import os,sys,json,runpy
forbidden=json.loads(sys.argv[1]);argv=sys.argv[2:]
def audit(event,args):
    if event in ('open','os.listdir','os.scandir') and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=os.path.realpath(os.fsdecode(args[0]))
        if any(path==root or path.startswith(root+os.sep) for root in forbidden):
            raise RuntimeError('Forbidden legacy directory access: '+path)
    if event=='import' and args and args[0].split('.')[0] in ('pyAgxArm','piper_sdk','pyrealsense2','show33'):
        raise RuntimeError('Hardware/team module imported in offline validation: '+args[0])
sys.addaudithook(audit)
if argv[0]=='-m':
    sys.argv=[argv[1],*argv[2:]];runpy.run_module(argv[1],run_name='__main__',alter_sys=True)
elif argv[0]=='-c':
    sys.argv=['-c',*argv[2:]];exec(compile(argv[1],'<audit-child>','exec'))
else:
    sys.argv=argv;runpy.run_path(argv[0],run_name='__main__')
"""
        return [sys.executable,'-c',bootstrap,json.dumps([str(p.resolve())for p in args.deny_path]),*argv]
    with tempfile.TemporaryDirectory(prefix='showonce-cpu-smoke-') as folder:
        root = Path(folder)
        config = root/'empty.toml'
        config.write_text('[paths]\nmodels = '+json.dumps(str(root/'models'))+'\nthird_party = '+json.dumps(str(root/'third_party'))+'\n')
        env = dict(os.environ, SHOWONCE_HOME=str(root/'data'), SHOWONCE_CONFIG=str(config),
                   PYTHONNOUSERSITE='1')
        env.pop('PYTHONPATH', None)
        generated = subprocess.run(python_command(str(repo/'examples/make_demo.py')),
                                   env=env, cwd=root, check=True, capture_output=True, text=True)
        # Core imports must come from the installed wheel rather than this checkout.
        location = subprocess.check_output(python_command('-c', 'import showonce_studio,piper_runtime;print(showonce_studio.__file__);print(piper_runtime.__file__)'),
            env=env, cwd=root, text=True).splitlines()
        if any(str(repo/'src') in path or str(repo.parent/'piper-runtime/src') in path for path in location):
            raise RuntimeError('Smoke test requires non-editable wheel installations')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        process = subprocess.Popen(python_command('-m', 'showonce_studio.showonce_studio','--offline', '--port', str(port)), env=env, cwd=root,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
        checked = []
        try:
            deadline = time.monotonic()+15
            while True:
                if process.poll() is not None:
                    raise RuntimeError(process.stderr.read().decode())
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=1) as response:
                        assert response.status == 200
                    break
                except urllib.error.URLError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Offline web startup timed out')
                    time.sleep(.1)
            for route in ('/', '/stages', '/objects', '/policies'):
                with urllib.request.urlopen(f'http://127.0.0.1:{port}{route}', timeout=2) as response:
                    assert response.status == 200
                    assert b'<html' in response.read().lower()
                checked.append(route)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            process.stderr.close()
        result = {'status': 'passed', 'synthetic_example': json.loads(generated.stdout),
                  'pages_checked': checked, 'installed_module_paths': location,
                  'offline_web_stopped': process.poll() is not None,
                  'camera_model_robot_workers_started': False, 'hardware_commands_sent': False,
                  'legacy_access_denied': bool(args.deny_path),
                  'denied_directories': [str(p.resolve())for p in args.deny_path]}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
