"""Native live perception contracts; no camera, SAM weights or SDK required."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from showonce_studio.perception.runner import DEFAULT_OBJECTS
from showonce_studio.perception.protocol import objects_document, provenance
from showonce_studio.perception.cube_image import cube_silhouette, estimate_cube, green_mask, CubeRejected
from showonce_studio.perception.bowl_image import silhouette, estimate_bowl


def test_native_import_does_not_load_external_team_code_or_devices(tmp_path):
    code = '''
import importlib.abc, sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'show33','live_shadow','pose_geom','pyrealsense2','pyAgxArm','piper_sdk','can','torch','sam3'}:
            raise AssertionError('Unexpected dependency: '+fullname)
sys.meta_path.insert(0,Deny())
from showonce_studio.perception import workers, protocol, runner
assert runner.DEFAULT_OBJECTS.is_file()
'''
    subprocess.run([sys.executable, '-c', code], cwd=tmp_path,
                   env=dict(os.environ, SHOWONCE_HOME=str(tmp_path)), check=True)


def test_worker_entry_preserves_reserved_cpu_affinity(tmp_path):
    cpu=min(os.sched_getaffinity(0))
    code='import os;import showonce_studio.showonce_live_vision;assert os.sched_getaffinity(0)=={'+str(cpu)+'}'
    subprocess.run([sys.executable,'-c',code],cwd=tmp_path,check=True,
        env=dict(os.environ,SHOWONCE_HOME=str(tmp_path),SHOW100_CPU_AFFINITY=str(cpu),
                 OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))


def scene():
    camera = np.array([.31, .608, .502])
    forward = np.array([.3, 0., 0.])-camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([right, np.cross(forward, right), forward])
    transform[:3, 3] = camera
    intrinsic = np.array([[620., 0, 424], [0, 620., 240], [0, 0, 1.]])
    return intrinsic, transform


@pytest.mark.parametrize('x,y,yaw',[(.25,-.1,-.6),(.3,0.,0.),(.38,.13,.35)])
def test_packaged_cube_recovers_geometry_without_surface_depth(x,y,yaw):
    import cv2
    definition = json.loads(DEFAULT_OBJECTS.read_text())['objects'][0]
    intrinsic, transform = scene(); normal = np.array([0.,0.,1.]); centre = np.array([x,y,.01])
    polygon = cube_silhouette(centre,yaw,np.full(3,.025),normal,intrinsic,transform)
    mask = np.zeros((480,848),np.uint8)
    cv2.fillConvexPoly(mask,np.rint(polygon).astype(np.int32),1)
    image = np.full((*mask.shape,3),[80,110,140],np.uint8);image[mask>0] = [100,180,15]
    pose, table, checks = estimate_cube(definition,mask>0,intrinsic,transform,
        table_normal=normal,table_offset=-.015,rgb=image)
    assert np.linalg.norm(pose[:3,3]-centre)<.002
    assert table == pytest.approx(-.015)
    assert checks['cube_surface_depth_used'] is False
    with pytest.raises(CubeRejected,match='colour_missing'):
        green_mask({'bgr':np.full_like(image,100)},mask>0)


def test_packaged_bowl_recovers_with_missing_bowl_depth():
    import cv2
    definition = json.loads(DEFAULT_OBJECTS.read_text())['objects'][1]
    intrinsic, transform = scene();centre = np.array([.3,0.,-.015]);normal = np.array([0.,0.,1.])
    yy,xx = np.indices((480,848))
    rays = np.stack([(xx-424)/620,(yy-240)/620,np.ones_like(xx)],-1)@transform[:3,:3].T
    depth = (-.015-transform[2,3])/rays[:,:,2]
    depth[(depth<.1)|(depth>2)] = 0
    polygon = silhouette(centre,normal,definition['size'],intrinsic,transform)
    mask = np.zeros(depth.shape,np.uint8);cv2.fillConvexPoly(mask,np.rint(polygon).astype(np.int32),1)
    depth[mask>0] = 0
    pose, _, checks = estimate_bowl(definition,mask,depth,intrinsic,transform,table_hint_z=-.015)
    assert np.linalg.norm(pose[:3,3]-centre)<.002
    assert checks['mask_depth_valid_fraction'] == 0.


def test_pose_file_contract_preserves_source_measurement_identity():
    assert len(provenance(DEFAULT_OBJECTS)['sha256']) == 64
    definitions = json.loads(DEFAULT_OBJECTS.read_text())['objects']
    transform = np.eye(4);transform[:3,3] = [.3,.1,.01]
    result = objects_document({'green_block':{'T':transform,'checks':{},'table_z_m':-.015,
        'frame':42,'t':123.5}},definitions,{'serial':'fixture'})
    item = result['objects'][0]
    assert (result['frame'],result['units']) == ('piper_base_link','m')
    assert item['source_frame'] == 42 and item['source_time_s'] == 123.5
    assert item['pos'] == [.3,.1,.01] and item['quat_wxyz'] == [1.,0.,0.,0.]


def test_service_command_uses_packaged_worker_without_legacy_paths(tmp_path,monkeypatch):
    from showonce_studio import live_vision_service as service
    monkeypatch.setattr(service,'ROOT',tmp_path)
    monkeypatch.setattr(service,'SETTINGS',{'live_vision':{'python':sys.executable,
        'inspector':'/missing/legacy.py','show33':'/missing/team-source'}})
    calibration = tmp_path/'calibration.json';calibration.write_text('{}')
    checkpoint = tmp_path/'sam.pt';checkpoint.write_bytes(b'fixture')
    monkeypatch.setattr(service,'sam_checkpoint',lambda:checkpoint)
    monkeypatch.setenv('PYTHONPATH','/app-env/site-packages:/legacy/source')
    manager = service.LiveVisionService(calibration,8768)
    assert manager.python==str(Path(sys.executable).absolute())
    assert manager.state()['configured'] and manager.state()['external_source_required'] is False
    class Planned(Exception):pass
    def spawn(command,**kwargs):
        assert 'PYTHONPATH' not in kwargs['env']
        assert '--show33' not in command and '--inspector' not in command
        assert 'showonce_studio.showonce_live_vision' in command
        assert '--objects' in command and '--checkpoint' in command
        raise Planned()
    monkeypatch.setattr(service.subprocess,'Popen',spawn)
    with pytest.raises(Planned):manager.ensure()


def test_service_keeps_venv_interpreter_symlink(tmp_path,monkeypatch):
    from showonce_studio import live_vision_service as service
    interpreter=tmp_path/'model-env/bin/python';interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    monkeypatch.setattr(service,'ROOT',tmp_path)
    monkeypatch.setattr(service,'SETTINGS',{'live_vision':{'python':str(interpreter)}})
    manager=service.LiveVisionService(tmp_path/'calibration.json',8768)
    assert manager.python==str(interpreter)
    assert manager.python!=str(interpreter.resolve())
