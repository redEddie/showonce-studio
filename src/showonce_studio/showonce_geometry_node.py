from showonce_studio.config import ROOT
#!/usr/bin/env python3
"""Persistent offline SAM/RGB-D analysis node. Never owns a camera or robot."""
import argparse
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from importlib import metadata
import os
from pathlib import Path
import threading
import time
import traceback
from urllib.parse import urlsplit, parse_qs
from showonce_studio.showonce_stages import read_json, write_json, recording_path
from showonce_studio.showonce_geometry import GeometryEngine, directory
from showonce_studio.showonce_relations import track_hands,directory as relation_directory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8769)
    parser.add_argument('--out',type=Path,default=ROOT/'datasets/demonstrations')
    args=parser.parse_args();engine=GeometryEngine();lock=threading.Lock();active={'id':None}

    def state(recording_id):
        recording,_=recording_path(args.out,recording_id);root=directory(recording)
        return {'recording_id':recording_id,'root':str(args.out.resolve()),'model_retained':engine.predictor is not None,
                'job':read_json(root/'job.json',{'status':'idle'}),'manifest':read_json(root/'manifest.json'),
                'tracking_ready':(root/'tracking.json').is_file(),
                'partial_tracking_ready':(root/'tracking.partial.json').is_file()}

    def work(recording_id):
        recording,_=recording_path(args.out,recording_id);root=directory(recording)
        job={'recording_id':recording_id,'pid':os.getpid(),'started_s':time.time()}
        def progress(status,**values):
            job.update(status=status,updated_s=time.time(),**values);write_json(root/'job.json',job)
            print(json.dumps(job,ensure_ascii=False),flush=True)
        try:
            with (ROOT/'runs/.showonce-vlm.lock').open('a') as gpu:
                while True:
                    try:fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                    except BlockingIOError:progress('waiting_for_gpu',message='다른 GPU 분석이 끝나기를 기다립니다.');time.sleep(1)
                tracking=engine.track(recording,progress)
            result=engine.reconstruct(recording,tracking,progress)
            progress('complete',message='객체 추적·다중 프레임 측정 표면 복원 완료',assets=len(result['assets']))
        except Exception as error:
            traceback.print_exc();progress('error',error=f'{type(error).__name__}: {error}')
        finally:
            with lock:active['id']=None

    def hand_state(recording_id):
        recording,_=recording_path(args.out,recording_id);root=relation_directory(recording)
        tracking=read_json(root/'hand_tracking.json')
        return {'recording_id':recording_id,'tracking_ready':bool(tracking and tracking.get('complete')),
                'model_retained':engine.predictor is not None,'job':read_json(root/'hand_job.json',{'status':'idle'})}

    def mask_state(recording_id):
        value=state(recording_id);recording,_=recording_path(args.out,recording_id)
        value['job']=read_json(directory(recording)/'mask_job.json',{'status':'idle'})
        return value

    def mask_work(recording_id):
        import torch
        recording,_=recording_path(args.out,recording_id);root=directory(recording)
        def progress(status,**values):write_json(root/'mask_job.json',{'status':status,'pid':os.getpid(),'updated_s':time.time(),**values})
        try:
            with (ROOT/'runs/.showonce-vlm.lock').open('a')as gpu:
                fcntl.flock(gpu,fcntl.LOCK_EX)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):engine.track(recording,progress)
            progress('complete',message='객체 마스크 추적 완료 · 형상 복원 없음')
        except Exception as error:traceback.print_exc();progress('error',error=f'{type(error).__name__}: {error}')
        finally:
            with lock:active['id']=None

    def hand_work(recording_id):
        import torch
        recording,_=recording_path(args.out,recording_id);root=relation_directory(recording)
        job={'recording_id':recording_id,'pid':os.getpid(),'started_s':time.time()}
        def progress(status,**values):
            job.update(status=status,updated_s=time.time(),**values);write_json(root/'hand_job.json',job)
        try:
            with (ROOT/'runs/.showonce-vlm.lock').open('a')as gpu:
                fcntl.flock(gpu,fcntl.LOCK_EX)
                # Autocast is thread-local: warm-up happens in another thread.
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    track_hands(recording,engine,progress)
            progress('complete',message='SAM 사람 손 추적 완료')
        except Exception as error:traceback.print_exc();progress('error',error=f'{type(error).__name__}: {error}')
        finally:
            with lock:active['id']=None

    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,value):
            body=json.dumps(value,ensure_ascii=False,allow_nan=False).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        def do_GET(self):
            try:
                if urlsplit(self.path).path=='/health':
                    return self.reply(200,{'root':str(args.out.resolve()),'model_retained':engine.predictor is not None,'active_recording_id':active['id'],
                        'environment':{'numpy':metadata.version('numpy'),'sam3':metadata.version('sam3')}})
                if urlsplit(self.path).path=='/api/geometry/hands/state':
                    return self.reply(200,hand_state(parse_qs(urlsplit(self.path).query).get('id',[''])[0]))
                if urlsplit(self.path).path=='/api/geometry/masks/state':
                    return self.reply(200,mask_state(parse_qs(urlsplit(self.path).query).get('id',[''])[0]))
                if urlsplit(self.path).path!='/api/geometry/state':return self.reply(404,{'error':'not found'})
                return self.reply(200,state(parse_qs(urlsplit(self.path).query).get('id',[''])[0]))
            except (ValueError,OSError) as error:self.reply(409,{'error':str(error)})
        def do_POST(self):
            try:
                if self.path not in ('/api/geometry/start','/api/geometry/hands/start','/api/geometry/masks/start'):return self.reply(404,{'error':'not found'})
                size=int(self.headers.get('Content-Length',0))
                if not 0<size<4096:raise ValueError('Invalid request size')
                body=json.loads(self.rfile.read(size))
                if not isinstance(body,dict):raise ValueError('JSON object required')
                recording_id=body.get('recording_id')
                if self.path=='/api/geometry/masks/start':
                    current=mask_state(recording_id)
                    if current['tracking_ready']:return self.reply(200,current)
                    with lock:
                        if active['id']is not None:raise ValueError('SAM 노드가 다른 분석을 진행 중입니다.')
                        active['id']=recording_id
                        threading.Thread(target=mask_work,args=(recording_id,),daemon=True).start()
                    return self.reply(202,mask_state(recording_id))
                if self.path=='/api/geometry/hands/start':
                    current=hand_state(recording_id)
                    if current['tracking_ready']:return self.reply(200,current)
                    with lock:
                        if active['id']is not None:raise ValueError('SAM 노드가 다른 분석을 진행 중입니다.')
                        active['id']=recording_id
                        recording,_=recording_path(args.out,recording_id)
                        write_json(relation_directory(recording)/'hand_job.json',{'status':'queued','started_s':time.time()})
                        threading.Thread(target=hand_work,args=(recording_id,),daemon=True).start()
                    return self.reply(202,hand_state(recording_id))
                current=state(recording_id)
                if current['manifest'] and current['manifest'].get('complete'):return self.reply(200,current)
                with lock:
                    if active['id'] not in (None,recording_id):raise ValueError('다른 영상의 형상을 분석 중입니다.')
                    if active['id'] is None:
                        active['id']=recording_id
                        recording,_=recording_path(args.out,recording_id)
                        write_json(directory(recording)/'job.json',{'status':'queued','started_s':time.time()})
                        threading.Thread(target=work,args=(recording_id,),daemon=True).start()
                return self.reply(202,state(recording_id))
            except (ValueError,OSError,TypeError) as error:self.reply(409,{'error':str(error)})
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    def warm_model():
        try:
            with (ROOT/'runs/.showonce-vlm.lock').open('a') as gpu:
                fcntl.flock(gpu,fcntl.LOCK_EX);engine.load()
            print('SAM model retained for future recordings',flush=True)
        except Exception:traceback.print_exc()
    threading.Thread(target=warm_model,daemon=True).start()
    print(f'ShowOnce geometry node: 127.0.0.1:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
