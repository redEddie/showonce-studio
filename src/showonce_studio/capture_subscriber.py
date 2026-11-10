"""Live RGB-D subscriber implementing the existing retained-camera protocol.

Reads one camera broker; never opens RealSense or a robot. Original source
timestamps, aligned depth, intrinsics, and frame identities are preserved.
"""
from collections import OrderedDict
import io
import json
import threading
import time
from types import SimpleNamespace
import urllib.request
from urllib.parse import urlencode
import numpy as np
from piper_policy_runtime.frame_retention import AnchorFrames


class CaptureSubscriber(threading.Thread):
    def __init__(self,url,serial,keep=20):
        super().__init__(daemon=True)
        self.url=url.rstrip('/');self.serial=str(serial);self.keep=keep
        self.frames=OrderedDict();self.lock=threading.Lock();self.new=threading.Condition()
        self.stop=False;self.latest=-1;self.K=None;self.domain='broker_host_wall_time'
        self.anchor_frames=AnchorFrames();self.missing_anchor_reads=0;self.stream_id=None;self.error=None
        self.pipe=SimpleNamespace(stop=lambda:setattr(self,'stop',True))

    def decode(self,payload):
        import cv2
        with np.load(io.BytesIO(payload),allow_pickle=False)as packet:
            bgr=packet['bgr'];z16=packet['depth_z16'];meta=json.loads(packet['metadata'].tobytes())
        camera=meta['camera'];stamps=meta['stamps'];stamp=float(stamps['host_wall_time_s'])
        if (meta.get('schema')!='showonce_live_rgbd_v1'or meta.get('offline_replay')is not False
                or meta.get('aligned_to')!='color'or str(camera['serial'])!=self.serial):
            raise ValueError('Live camera broker identity/alignment mismatch')
        if not -.01<=time.time()-stamp<=.5:raise ValueError('Live camera broker frame expired')
        if self.stream_id is not None and self.stream_id!=meta['stream_id']:raise ValueError('Camera broker restarted; restart perception to reset frame identities')
        self.stream_id=meta['stream_id'];intr=camera['intrinsics']
        shape=(intr['height'],intr['width'])
        if bgr.shape!=(*shape,3)or bgr.dtype!=np.uint8 or z16.shape!=shape or z16.dtype!=np.uint16:
            raise ValueError('Invalid aligned RGB-D dimensions/dtypes')
        self.K=np.array([[intr['fx'],0.,intr['cx']],[0.,intr['fy'],intr['cy']],[0.,0.,1.]])
        depth=z16.astype(np.float32)*float(camera['depth_scale_m'])
        if not np.isfinite(self.K).all()or not np.isfinite(depth).all():raise ValueError('Invalid RGB-D camera parameters')
        return dict(id=int(meta['sequence']),bgr=bgr,hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV),depth=depth,
            t=stamp,t_arrive=stamp,t_aligned=stamp,t_ready=time.time())

    def run(self):
        try:
            while not self.stop:
                request=self.url+'/api/capture/rgbd?'+urlencode({'after':self.latest})
                with urllib.request.urlopen(request,timeout=2.)as response:frame=self.decode(response.read(4*1024*1024))
                if frame['id']==self.latest:continue
                with self.lock:
                    self.frames[frame['id']]=frame
                    while len(self.frames)>self.keep:self.frames.popitem(last=False)
                    self.latest=frame['id']
                with self.new:self.new.notify_all()
        except Exception as error:
            if not self.stop:self.error=f'{type(error).__name__}: {error}'

    def get(self,frame_id=None):
        with self.lock:frame=self.frames.get(self.latest if frame_id is None else frame_id)
        if frame is None and frame_id is not None:
            frame=self.anchor_frames.get(frame_id)
            if frame is None:self.missing_anchor_reads+=1
        return frame

    def wait_newer(self,last,timeout=1.):
        with self.new:self.new.wait_for(lambda:self.latest>last or self.stop or self.error,timeout)
        return self.get()
