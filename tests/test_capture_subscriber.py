import io
import json
import time
import numpy as np
import pytest
from showonce_studio.showonce_capture_node import Capture
from showonce_studio.capture_subscriber import CaptureSubscriber


def broker(tmp_path):
    capture=Capture(tmp_path,tmp_path/'calibration.json')
    capture.camera={'serial':'fixture','depth_scale_m':.001,'intrinsics':
        {'width':4,'height':3,'fx':10.,'fy':11.,'cx':2.,'cy':1.}}
    color=np.full((3,4,3),80,np.uint8);depth=np.full((3,4),500,np.uint16)
    stamp=time.time();capture.publish_rgbd(color,depth,{'host_wall_time_s':stamp})
    return capture,color,depth,stamp


def test_broker_owns_buffers_and_subscriber_preserves_source_time_intrinsics_depth(tmp_path):
    capture,color,depth,stamp=broker(tmp_path);color[:]=0;depth[:]=0
    subscriber=CaptureSubscriber('http://127.0.0.1:1','fixture')
    frame=subscriber.decode(capture.rgbd())
    assert frame['t']==stamp and frame['id']==1
    assert np.all(frame['bgr']==80)
    np.testing.assert_allclose(frame['depth'],.5)
    np.testing.assert_equal(subscriber.K,[[10.,0.,2.],[0.,11.,1.],[0.,0.,1.]])
    assert not capture.thread.is_alive()and not subscriber.is_alive()


def test_expired_wrong_camera_and_camera_restart_fail_explicitly(tmp_path):
    capture,_,_,_=broker(tmp_path)
    with pytest.raises(ValueError,match='identity'):CaptureSubscriber('http://127.0.0.1:1','other').decode(capture.rgbd())
    subscriber=CaptureSubscriber('http://127.0.0.1:1','fixture');subscriber.decode(capture.rgbd())
    capture.rgbd_stream_id='new-camera-stream'
    with pytest.raises(ValueError,match='restarted'):subscriber.decode(capture.rgbd())
    capture.publish_rgbd(np.zeros((3,4,3),np.uint8),np.zeros((3,4),np.uint16),{'host_wall_time_s':time.time()-2})
    with pytest.raises(ValueError,match='만료'):capture.rgbd()
