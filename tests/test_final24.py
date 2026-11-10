"""Pinned release ingestion and immutable parent/final model separation."""
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
import pytest
from showonce_studio.policy_release import install,presentation,release_spec,verify_archive
from showonce_studio.showonce_policies import PolicyLibrary,sha

def test_managed_policy_command_ignores_legacy_file_shadowing(tmp_path):
    from showonce_studio.robot_actions import RobotActions
    from showonce_studio.config import runtime_env
    (tmp_path/'showonce_studio.py').write_text('raise AssertionError("legacy file must not shadow installed package")')
    manager=RobotActions(tmp_path,PolicyLibrary(tmp_path/'library'),jobs=tmp_path/'jobs',offline=True)
    assert '-P' in manager.policy_command and '-P' in manager.sdk_command
    result=subprocess.run([*manager.policy_command,'--help'],cwd=tmp_path,env=runtime_env(),capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '--policy-id' in result.stdout

def test_parent_or_mixed_normalizer_cannot_receive_final_model_badge():
    s=release_spec()
    meta=dict(checkpoint_sha256=s['checkpoint_sha256'],weights_sha256=s['weights_sha256'],training_config_sha256=s['config_sha256'])
    assert presentation(meta)['is_final']
    for key in meta:
        changed=copy.deepcopy(meta);changed[key]='parent'
        assert presentation(changed)=={}

def test_corrupt_archive_is_rejected_before_registration(tmp_path):
    archive=tmp_path/'bad.tar.gz';archive.write_bytes(b'not the selected archive')
    with pytest.raises(ValueError,match='SHA256'):verify_archive(archive,release_spec())
    assert not (tmp_path/'library').exists()

def test_actual_final_archive_import_is_immutable_and_keeps_recorded_vectors(tmp_path):
    archive=os.environ.get('PIPER_FINAL24_ARCHIVE')
    if not archive:pytest.skip('Set PIPER_FINAL24_ARCHIVE to the official inference.tar.gz')
    library=PolicyLibrary(tmp_path/'library')
    result=install(archive,library,artifacts=tmp_path/'artifacts');item=result['policy']
    assert item['is_final'] and item['training_step']==201326592
    assert item['hardware_validated'] is False and result['hardware_commands_sent'] is False
    assert result['validation']['recorded_replay']['steps']==200
    registered=library.folder(item['id']);assert (registered/'recorded-golden.npz').is_file()
    assert sha(registered/'weights.npz')==release_spec()['weights_sha256']
    repeated=install(archive,library,artifacts=tmp_path/'artifacts')
    assert repeated['policy']==item
    assert len(library.catalog())==1
    meta=json.loads((registered/'policy.json').read_text())
    assert meta['checkpoint_sha256']==release_spec()['checkpoint_sha256']
