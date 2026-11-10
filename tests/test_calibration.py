"""No camera, SDK or CUDA is needed for these contract checks."""
import ast
import hashlib
import json
from pathlib import Path
import socket
import struct

import numpy as np
import pytest

from showonce_studio.calibration.cli import check_candidate, check_feedback, main
from showonce_studio.calibration.common import backproject, transform, inv_T
from showonce_studio.calibration.capture import frame_metadata
from showonce_studio.calibration import cli, snapshot


def feedback():
    return dict(telemetry_fresh_at_frame=True,q_rad=[0.]*6,qvel_rad_s=[0.]*6,gripper_full_width_m=.1)


@pytest.mark.parametrize('update',[
    {'telemetry_fresh_at_frame':False},{'q_rad':[0]*5}, {'q_rad':[float('nan')]*6},
    {'qvel_rad_s':[.021]*6},{'qvel_rad_s':[None]*6}, {'gripper_full_width_m':None},
    {'gripper_full_width_m':float('inf')}, {'gripper_full_width_m':-.001}])
def test_capture_requires_fresh_stationary_feedback(update):
    f=feedback();f.update(update)
    with pytest.raises(ValueError):check_feedback(f)


def test_legacy_candidate_thresholds():
    cal=dict(trusted=True,serial='123',table_check=dict(height_m=-.0136,tilt_deg=.28,plane_rms_mm=1.99,n_points=25168))
    checks,passed=check_candidate(cal,'123',-.015)
    assert passed and checks['table_height_difference_mm']==1.4
    assert not check_candidate(cal,'other',-.015)[1]
    cal['table_check']['height_m']=.05
    assert not check_candidate(cal,'123',-.015)[1]


def test_geometry_units_and_frame():
    depth=np.array([[0.,1.],[2.,0.]])
    k=np.array([[2.,0,0],[0,2.,0],[0,0,1]])
    points=backproject(depth,k)
    np.testing.assert_allclose(points,[[.5,0,1],[0,1,2]])
    t=np.eye(4);t[:3,3]=[.1,.2,-.015]
    np.testing.assert_allclose(transform(inv_T(t),transform(t,points)),points,atol=1e-15)


def test_robot_assets_are_packaged_and_byte_preserved():
    root=Path(__file__).parents[1]
    manifest=json.loads((root/'provenance/camera-calibration.json').read_text())
    count=0
    for row in manifest['files']:
        path=root/row['destination']
        if '/data/' in row['destination']:
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
            count+=1
    assert count>50


def test_no_external_team_imports_or_command_transmission():
    folder=Path(cli.__file__).parent
    for path in folder.glob('*.py'):
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):assert not (node.module or '').startswith(('show33','piper_sdk','pyAgxArm'))
            if isinstance(node,ast.Import):assert all(not a.name.startswith(('show33','piper_sdk','pyAgxArm')) for a in node.names)
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                assert node.func.attr not in {'send','sendall','sendto','EnablePiper','GripperCtrl','JointCtrl'}
    assert 'socket.socket' not in (folder/'capture.py').read_text() # shared Runtime RX only


def test_offline_failure_keeps_active_calibration_and_writes_summary(tmp_path):
    src=tmp_path/'camera';src.mkdir()
    (src/'meta.json').write_text('{"serial":"123"}')
    fb=tmp_path/'feedback.json';fb.write_text(json.dumps(dict(feedback(),telemetry_fresh_at_frame=False)))
    active=tmp_path/'current.json';active.write_text('{"serial":"123", "trusted":true}')
    original=active.read_bytes();out=tmp_path/'candidate'
    assert main(['--src',str(src),'--feedback',str(fb),'--out',str(out),'--calib',str(active)])==1
    report=json.loads((out/'summary.json').read_text())
    assert not report['hardware_commands_sent'] and not report['active_calibration_replaced']
    assert active.read_bytes()==original


def test_registration_worker_uses_own_environment(monkeypatch):
    monkeypatch.setenv('PYTHONPATH','/external/app/numpy2')
    calls=[]
    class Result:
        returncode=0;stdout='ok';stderr=''
    monkeypatch.setattr(snapshot.subprocess,'run',lambda cmd,**kw: calls.append((cmd,kw)) or Result())
    assert snapshot.run(['/worker/bin/python','-m','showonce_studio.calibration.register'])=='ok'
    assert 'PYTHONPATH' not in calls[0][1]['env']
    assert calls[0][1]['env']['PATH'].startswith('/worker/bin:')


def test_flat_studio_file_cannot_shadow_worker_package(tmp_path,monkeypatch):
    import sys
    (tmp_path/'showonce_studio.py').write_text("raise RuntimeError('flat module was imported')")
    monkeypatch.chdir(tmp_path)
    output=snapshot.run([sys.executable,'-m','showonce_studio.calibration.cli','--help'])
    assert '--robot-box' in output and '--mount' in output
