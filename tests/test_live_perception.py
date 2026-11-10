import time
from pathlib import Path
from unittest.mock import patch
import pytest
from showonce_studio.live_perception import LivePerception
from showonce_studio.showonce_stages import write_json


def service(tmp_path):
    folder=tmp_path/'runs/show100-vision-current';folder.mkdir(parents=True)
    write_json(folder/'manifest.json',{'mode':'live_camera_only','calibration':{'path':'/calibration.json'}})
    write_json(folder/'tracked_poses.json',{'offline_replay':False,'sample_time_s':time.time(),
        'objects':[{'id':'green_block','pos':[.3,.1,.01],'source_time_s':time.time()-.05,'estimator':'RGB_known_cube_support_plane'},
                   {'id':'bowl','pos':[.3,-.15,-.015],'source_time_s':time.time()-.7}]})
    (folder/'latest.jpg').write_bytes(b'jpeg-fixture')
    return folder,LivePerception(tmp_path)


def test_observed_age_is_not_replaced_by_new_heartbeat(tmp_path):
    folder,monitor=service(tmp_path)
    with patch('subprocess.Popen')as spawn:
        state=monitor.state();spawn.assert_not_called()
    assert state['connected']
    assert [o['status']for o in state['objects']]==['fresh','delayed']
    assert .6<state['objects'][1]['observation_age_s']<1.
    assert state['preview_kind']=='perception_overlay'
    assert monitor.frame(str(folder))==(b'jpeg-fixture','image/jpeg')
    stale=monitor.state(now=time.time()+5)
    assert not stale['connected']
    assert all(o['status']=='missing'for o in stale['objects'])
    assert stale['preview_kind']=='capture_only'


def test_empty_does_not_present_capture_as_sam_detection(tmp_path):
    state=LivePerception(tmp_path).state()
    assert not state['connected']and state['objects']==[]
    assert state['preview_kind']=='capture_only'and 'SAM 검출 결과가 아닙니다'in state['message']


def test_recordings_path_escape_and_old_preview_rejected(tmp_path):
    folder,monitor=service(tmp_path)
    for path in (tmp_path,tmp_path/'runs/../private',tmp_path/'recordings'):
        with pytest.raises(ValueError):monitor.state(str(path))
    write_json(folder/'manifest.json',{'mode':'archived_rgbd_replay'})
    with pytest.raises(ValueError):monitor.frame(str(folder))
    write_json(folder/'manifest.json',{'mode':'live_camera_only'})
    write_json(folder/'tracked_poses.json',{'offline_replay':True,'sample_time_s':time.time()})
    assert not monitor.state(str(folder))['connected']
    with pytest.raises(ValueError):monitor.frame(str(folder))


def test_frame_symlink_cannot_expose_another_file(tmp_path):
    folder,monitor=service(tmp_path);(folder/'latest.jpg').unlink()
    secret=tmp_path/'other.jpg';secret.write_bytes(b'other');(folder/'latest.jpg').symlink_to(secret)
    with pytest.raises(ValueError):monitor.frame(str(folder))
