from showonce_studio.config import ROOT
#!/usr/bin/env python3
"""Headless MuJoCo scene runtime for the web preview; no camera or robot SDK."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit,parse_qs
import numpy as np
import mujoco
from showonce_studio.showonce_stages import recording_path,read_json
from showonce_studio.showonce_sim_assets import directory


class SceneWorld:
    def __init__(self,root):
        self.root=Path(root);self.lock=threading.RLock();self.model=None;self.data=None
        self.recording_id=None;self.scenario='initial';self.running=False;self.last_seen=time.monotonic()
        self.ids={};self.accumulator=0.;self.pause_reason=None

    def load(self,recording_id,scenario='initial',force=False):
        if scenario not in ('initial','drop'):raise ValueError('Unknown simulation scenario')
        recording,_=recording_path(self.root,recording_id);folder=directory(recording)
        manifest=read_json(folder/'manifest.json',{})
        if not manifest.get('simulation_ready'):raise ValueError('물리 검증을 통과한 시뮬레이션 에셋이 필요합니다.')
        with self.lock:
            if not force and self.recording_id==recording_id and self.scenario==scenario:return self.state(recording_id)
            model=mujoco.MjModel.from_xml_path(str((folder/('scene.xml' if scenario=='initial' else 'container-check.xml')).resolve()))
            ids={asset['id']:model.body(asset['id']).id for asset in manifest['assets']}
            data=mujoco.MjData(model);mujoco.mj_forward(model,data)
            self.model,self.data,self.ids=model,data,ids
            self.recording_id=recording_id;self.scenario=scenario;self.running=False
            self.accumulator=0.;self.pause_reason=None;self.last_seen=time.monotonic()
            return self.state(recording_id)

    def state(self,recording_id=None):
        with self.lock:
            if self.model is None or (recording_id is not None and self.recording_id!=recording_id):
                return {'loaded':False,'recording_id':recording_id,'active_recording_id':self.recording_id}
            self.last_seen=time.monotonic()
            bodies={body:oid for oid,body in self.ids.items()};pairs={}
            for contact in self.data.contact:
                if contact.dist>.001:continue
                entities=[]
                for geom in contact.geom:
                    name=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,int(geom))
                    entities.append('table'if name=='table'else bodies.get(int(self.model.geom_bodyid[geom])))
                if None in entities or entities[0]==entities[1]:continue
                key=tuple(sorted(entities));pairs[key]=pairs.get(key,0)+1
            return {'loaded':True,'backend':'MuJoCo '+mujoco.__version__,'recording_id':self.recording_id,
                'scenario':self.scenario,'running':self.running,'time_s':float(self.data.time),
                'dt_s':float(self.model.opt.timestep),'contacts':int(self.data.ncon),'pause_reason':self.pause_reason,
                'contact_pairs':[{'a':a,'b':b,'count':count}for (a,b),count in pairs.items()],
                'objects':{oid:{'position':self.data.xpos[body].tolist(),'quat_wxyz':self.data.xquat[body].tolist()}
                           for oid,body in self.ids.items()}}

    def command(self,recording_id,action):
        if action not in ('load','play','pause','reset','drop'):raise ValueError('Unknown scene action')
        if action=='load':return self.load(recording_id)
        if action in ('reset','drop'):
            self.load(recording_id,'drop' if action=='drop' else 'initial',force=True)
            with self.lock:self.running=action=='drop'
            return self.state(recording_id)
        with self.lock:
            if self.model is None or self.recording_id!=recording_id:raise ValueError('먼저 해당 시연의 시뮬레이션 장면을 여세요.')
            self.running=action=='play';self.pause_reason=None;self.last_seen=time.monotonic()
            return self.state(recording_id)

    def advance(self,elapsed):
        with self.lock:
            if self.model is None or not self.running:return
            self.accumulator+=min(max(0,elapsed),.1)
            steps=int(self.accumulator/self.model.opt.timestep)
            if not steps:return
            self.accumulator-=steps*self.model.opt.timestep
            mujoco.mj_step(self.model,self.data,nstep=steps);mujoco.mj_forward(self.model,self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                self.running=False;self.pause_reason='물리 상태를 확인해야 합니다.'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8770)
    parser.add_argument('--out',type=Path,default=ROOT/'datasets/demonstrations')
    args=parser.parse_args();world=SceneWorld(args.out)
    class Handler(BaseHTTPRequestHandler):
        def reply(self,code,value):
            body=json.dumps(value,ensure_ascii=False,allow_nan=False).encode();self.send_response(code)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        def do_GET(self):
            try:
                path=urlsplit(self.path).path
                if path=='/health':return self.reply(200,{'root':str(args.out.resolve()),'backend':'MuJoCo '+mujoco.__version__})
                if path!='/state':return self.reply(404,{'error':'not found'})
                recording_id=parse_qs(urlsplit(self.path).query).get('id',[''])[0]
                recording_path(args.out,recording_id)
                return self.reply(200,world.state(recording_id))
            except (ValueError,OSError,KeyError,TypeError) as error:self.reply(409,{'error':str(error)})
        def do_POST(self):
            try:
                if self.path!='/command':return self.reply(404,{'error':'not found'})
                size=int(self.headers.get('Content-Length',0))
                if not 0<size<4096:raise ValueError('Invalid request size')
                body=json.loads(self.rfile.read(size))
                if not isinstance(body,dict):raise ValueError('JSON object required')
                recording_id=body.get('recording_id');recording_path(args.out,recording_id)
                return self.reply(200,world.command(recording_id,body.get('action')))
            except (ValueError,OSError,KeyError,TypeError) as error:self.reply(409,{'error':str(error)})
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    def run():
        previous=time.monotonic()
        while True:
            now=time.monotonic();elapsed=now-previous;previous=now
            with world.lock:
                if world.running and now-world.last_seen>3:
                    world.running=False;world.pause_reason='화면 연결이 없어 일시정지했습니다.'
            world.advance(elapsed);time.sleep(.004)
    threading.Thread(target=run,daemon=True).start()
    print(f'ShowOnce MuJoCo scene: 127.0.0.1:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
