"""Build the required SAM worker in an isolated Thor environment; no model/device startup."""
import argparse
from pathlib import Path
import platform
import subprocess
import sys

SOURCE = 'https://github.com/facebookresearch/sam3.git'
REVISION = '7f473cfd983d5eb43040ff608f66bff0d7bdcb53'


def run(args, **kwargs):
    subprocess.run([str(value) for value in args], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--venv', type=Path, default=repo/'.venv-sam')
    parser.add_argument('--source', type=Path, default=repo/'.workspace/sam3-upstream')
    parser.add_argument('--runtime', type=Path, default=repo.parent/'piper-runtime')
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        parser.error('This recipe is validated for Linux aarch64/Thor; use a separately tested profile elsewhere.')
    version = subprocess.check_output([args.python, '-c', 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")'], text=True).strip()
    if version != '3.12':
        parser.error('Use --python with a Python 3.12 interpreter.')
    args.venv = args.venv.expanduser().resolve()
    args.source = args.source.expanduser().resolve()
    args.runtime = args.runtime.expanduser().resolve()
    if not (args.runtime/'pyproject.toml').is_file():
        parser.error('A sibling piper-runtime checkout is required.')
    if not (args.venv/'pyvenv.cfg').is_file():
        if args.venv.exists():
            parser.error('The target exists and is not a venv; choose an empty target.')
        run([args.python, '-m', 'venv', args.venv])
    if 'include-system-site-packages = false' not in (args.venv/'pyvenv.cfg').read_text().lower():
        parser.error('The worker venv must not inherit system site-packages.')
    python = args.venv/'bin/python'
    # A dedicated model environment must not silently downgrade an existing NumPy 2 deployment.
    installed_numpy = subprocess.run([str(python), '-c', 'import numpy;print(numpy.__version__)'], capture_output=True, text=True)
    if installed_numpy.returncode == 0 and not installed_numpy.stdout.strip().startswith('1.26.'):
        parser.error('This venv already uses another NumPy version; choose a fresh dedicated SAM venv.')
    run([python, '-m', 'pip', 'install', 'pip==26.3', 'wheel==0.48.0', 'numpy==1.26.4'])
    if not args.source.exists():
        args.source.parent.mkdir(parents=True, exist_ok=True)
        run(['git', 'clone', '--filter=blob:none', '--no-checkout', SOURCE, args.source])
        run(['git', 'checkout', '--detach', REVISION], cwd=args.source)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.source, text=True).strip()
    if revision != REVISION:
        parser.error('The SAM source revision differs; choose a new source directory.')
    if subprocess.check_output(['git', 'diff', 'HEAD'], cwd=args.source):
        parser.error('Original SAM source must be unmodified; choose a clean source directory.')
    run([python, '-m', 'pip', 'install', 'torch==2.10.0+cu130', 'torchvision==0.25.0+cu130',
         '--index-url', 'https://download.pytorch.org/whl/cu130'])
    run([python, '-m', 'pip', 'install', '-c', repo/'requirements/sam-worker-thor-cu130.txt',
         args.source, args.runtime, f'{repo}[perception]',
         '--report', args.venv/'install-report.json'])
    run([python, repo/'scripts/repair_cusparselt_tag.py', '--wheelhouse', args.venv/'.wheelhouse'])
    run([python, '-m', 'pip', 'check'])
    run([python, '-m', 'piper_runtime.identity', '--lock', repo/'piper-runtime.lock.json'])
    print('SAM worker installed. Use scripts/verify_sam_worker.py with approved weights and recorded RGB frames.')


if __name__ == '__main__':
    main()
