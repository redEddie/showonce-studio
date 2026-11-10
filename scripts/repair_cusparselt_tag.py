"""Correct a known NVIDIA ARM64 wheel metadata tag, preserving every binary byte."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheelhouse', type=Path, required=True)
    args = parser.parse_args()
    dist = metadata.distribution('nvidia-cusparselt-cu13')
    text = dist.read_text('WHEEL')
    if 'Tag: py3-none-manylinux2014_aarch64' in text:
        print('cuSPARSELt metadata already has the correct aarch64 tag.')
        return
    if dist.version != '0.8.0' or platform.machine() != 'aarch64' or 'Tag: py3-none-manylinux2014_sbsa' not in text:
        raise RuntimeError('Unexpected wheel/platform; this targeted repair does not apply.')
    args.wheelhouse.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-m', 'pip', 'download', '--no-deps',
                    '--index-url', 'https://download.pytorch.org/whl/cu130',
                    '--dest', str(args.wheelhouse), 'nvidia-cusparselt-cu13==0.8.0'], check=True)
    original = args.wheelhouse/'nvidia_cusparselt_cu13-0.8.0-py3-none-manylinux2014_aarch64.whl'
    if sha(original) != '400c6ed1cf6780fc6efedd64ec9f1345871767e6a1a0a552a1ea0578117ea77c':
        raise RuntimeError('The upstream wheel hash differs from the verified artifact; review before repairing.')
    # The filename is aarch64 but the official archive's WHEEL metadata says sbsa.
    # Validate actual ELF headers before aligning the filename for wheel's tag tool.
    with zipfile.ZipFile(original) as source:
        libraries = [name for name in source.namelist() if '.so' in Path(name).name]
        if not libraries:
            raise RuntimeError('No shared library in the expected NVIDIA archive.')
        for name in libraries:
            with source.open(name) as stream:
                header = stream.read(20)
            if header[:6] != b'\x7fELF\x02\x01' or int.from_bytes(header[18:20], 'little') != 183:
                raise RuntimeError('Expected little-endian ARM64 ELF; no repair performed.')
    aligned = args.wheelhouse/'nvidia_cusparselt_cu13-0.8.0-py3-none-manylinux2014_sbsa.whl'
    shutil.copyfile(original, aligned)
    subprocess.run([sys.executable, '-m', 'wheel', 'tags', '--platform-tag=manylinux2014_aarch64',
                    '--build=1', str(aligned)], check=True)
    corrected = args.wheelhouse/'nvidia_cusparselt_cu13-0.8.0-1-py3-none-manylinux2014_aarch64.whl'
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(corrected) as target:
        if set(source.namelist()) != set(target.namelist()):
            raise RuntimeError('Unexpected archive contents after retagging.')
        preserved = []
        for name in source.namelist():
            if name.endswith(('.dist-info/WHEEL', '.dist-info/RECORD')):
                continue
            # Digest all upstream contents, including libraries and license files.
            a = hashlib.sha256(source.read(name)).hexdigest()
            b = hashlib.sha256(target.read(name)).hexdigest()
            if a != b:
                raise RuntimeError('Retagging changed upstream content: '+name)
            preserved.append(name)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--force-reinstall', str(corrected)], check=True)
    report = {'source': 'Official PyTorch cu130 index / NVIDIA wheel', 'version': '0.8.0',
              'original_sha256': sha(original), 'corrected_sha256': sha(corrected),
              'original_metadata_tag': 'py3-none-manylinux2014_sbsa',
              'corrected_metadata_tag': 'py3-none-manylinux2014_aarch64',
              'elf_machine': 'AArch64 (183)', 'preserved_files': len(preserved),
              'binary_and_license_contents_unchanged': True,
              'note': '--no-deps is used only to download one artifact; installation uses normal dependency checking.'}
    (args.wheelhouse/'cusparselt-tag-repair.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
