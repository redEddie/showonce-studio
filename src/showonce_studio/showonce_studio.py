from showonce_studio.config import service_port
from showonce_studio.config import WEB_ROOT
from showonce_studio.config import default_calibration
#!/usr/bin/env python3
"""ShowOnce Studio capture, VLM stage review and separate camera node launcher."""
import argparse
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from urllib.parse import parse_qs, urlsplit
from showonce_studio.config import PYTHON, runtime_env
from showonce_studio.showonce_stages import StageManager, recording_path, read_json
from showonce_studio.showonce_models import DEFAULT_MODEL, MODELS, model_status
from showonce_studio.showonce_geometry_api import GeometryManager
from showonce_studio.showonce_sim_assets_api import SimulationAssetsManager
from showonce_studio.showonce_scene_api import SceneManager
from showonce_studio.showonce_replay import ReplayManager
from showonce_studio.showonce_pose_api import PoseManager
from showonce_studio.showonce_relations_api import RelationsManager
from showonce_studio.showonce_library_api import LibraryManager
from showonce_studio.showonce_deployment_api import DeploymentManager
from showonce_studio.robot_actions import RobotActions
from showonce_studio.live_vision_service import LiveVisionService

from showonce_studio.config import ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--offline',action='store_true',help='Read local artifacts without starting a camera or model-analysis worker')
    ap.add_argument('--port', type=int, default=service_port('web',8765))
    ap.add_argument('--node-port', type=int, default=service_port('capture',8768))
    ap.add_argument('--out', type=Path, default=ROOT/'datasets/demonstrations')
    ap.add_argument('--calib', type=Path, default=default_calibration())
    args = ap.parse_args()
    live_vision=LiveVisionService(args.calib,args.node_port)
    child = None
    stages = StageManager(args.out)
    geometry = GeometryManager(args.out)
    simulation = SimulationAssetsManager(args.out)
    scenes = SceneManager(args.out)
    replays = ReplayManager(args.out)
    poses = PoseManager(args.out)
    relations = RelationsManager(args.out)
    library = LibraryManager(args.out)
    deployment = DeploymentManager(args.out)
    robot_actions = RobotActions(args.out,deployment.library,offline=args.offline,calibration=args.calib)

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, mime, encoding=None):
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            if encoding:self.send_header('Content-Encoding',encoding)
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split('?')[0]
            if path == '/':
                return self.send(200, (WEB_ROOT/'showonce_capture.html').read_bytes(), 'text/html; charset=utf-8')
            if path == '/stages':
                return self.send(200, (WEB_ROOT/'showonce_stages.html').read_bytes(), 'text/html; charset=utf-8')
            if path == '/objects':
                return self.send(200,(WEB_ROOT/'showonce_objects.html').read_bytes(),'text/html; charset=utf-8')
            if path in ('/showonce_workflow.css','/showonce_stages_ui.js','/showonce_objects_ui.js'):
                mime='text/css; charset=utf-8'if path.endswith('.css')else 'text/javascript; charset=utf-8'
                return self.send(200,(WEB_ROOT/path.lstrip('/')).read_bytes(),mime)
            if path == '/policies':
                return self.send(200,(WEB_ROOT/'showonce_policies.html').read_bytes(),'text/html; charset=utf-8')
            if path in ('/showonce_policies_ui.js','/showonce_robot_ui.js','/showonce_live_ui.js','/showonce_xy_ui.js'):
                return self.send(200,(WEB_ROOT/path.lstrip('/')).read_bytes(),'text/javascript; charset=utf-8')
            if path.startswith('/api/robot/'):
                return self.robot_api(path)
            if path.startswith('/api/deployment/'):
                return self.deployment_api(path)
            if path == '/showonce_geometry_ui.js':
                return self.send(200,(WEB_ROOT/'showonce_geometry_ui.js').read_bytes(),'text/javascript; charset=utf-8')
            if path == '/showonce_scene_ui.js':
                return self.send(200,(WEB_ROOT/'showonce_scene_ui.js').read_bytes(),'text/javascript; charset=utf-8')
            if path == '/showonce_replay_math.js':
                return self.send(200,(WEB_ROOT/'showonce_replay_math.js').read_bytes(),'text/javascript; charset=utf-8')
            if path == '/showonce_relations_ui.js':
                return self.send(200,(WEB_ROOT/'showonce_relations_ui.js').read_bytes(),'text/javascript; charset=utf-8')
            if path == '/showonce_library_ui.js':
                return self.send(200,(WEB_ROOT/'showonce_library_ui.js').read_bytes(),'text/javascript; charset=utf-8')
            if path.startswith('/api/library/'):
                return self.library_api(path)
            match=re.fullmatch(r'/library/([a-z0-9-]+)/(v[1-9][0-9]*)/(.*)',path)
            if match:
                try:
                    file=library.library.artifact(*match.groups())
                    mime={'.glb':'model/gltf-binary','.json':'application/json','.png':'image/png'}.get(file.suffix,'application/octet-stream')
                    return self.send(200,file.read_bytes(),mime)
                except (OSError,ValueError):return self.send(404,b'Asset unavailable','text/plain')
            if path == '/vendor/echarts.min.js':
                return self.send(200,(WEB_ROOT/'vendor/echarts.min.js').read_bytes(),'text/javascript; charset=utf-8')
            if path.startswith('/api/relations/'):
                return self.relations_api(path)
            if path.startswith('/vendor/three/'):
                file=(WEB_ROOT/path.lstrip('/')).resolve()
                if (WEB_ROOT/'vendor/three').resolve() not in file.parents or not file.is_file():
                    return self.send(404,b'Not found','text/plain')
                return self.send(200,file.read_bytes(),'text/javascript' if file.suffix=='.js' else 'text/plain')
            if path.startswith('/api/geometry/'):
                return self.geometry_api(path)
            if path.startswith('/api/pose/'):
                return self.pose_api(path)
            match=re.fullmatch(r'/poses/([0-9-]+)/(.*)',path)
            if match:
                try:return self.send(200,poses.artifact(match[1],match[2]).read_bytes(),'application/json')
                except (OSError,ValueError):return self.send(404,b'Artifact unavailable','text/plain')
            if path.startswith('/api/simulation/'):
                return self.simulation_api(path)
            if path.startswith('/api/scene/'):
                return self.scene_api(path)
            match=re.fullmatch(r'/simulation/([0-9-]+)/(.*)',path)
            if match:
                try:
                    file=simulation.artifact(match[1],match[2])
                    mime={'.glb':'model/gltf-binary','.json':'application/json','.png':'image/png','.zip':'application/zip'}.get(file.suffix,'application/octet-stream')
                    return self.send(200,file.read_bytes(),mime)
                except (OSError,ValueError):return self.send(404,b'Artifact unavailable','text/plain')
            match=re.fullmatch(r'/geometry/([0-9-]+)/(.*)',path)
            if match:
                try:
                    file=geometry.artifact(match[1],match[2])
                    mime={'.glb':'model/gltf-binary','.json':'application/json','.png':'image/png'}.get(file.suffix,'application/octet-stream')
                    return self.send(200,file.read_bytes(),mime)
                except (OSError,ValueError):return self.send(404,b'Artifact unavailable','text/plain')
            if path.startswith('/api/stages/'):
                return self.stage_api(path)
            match = re.fullmatch(r'/recordings/([0-9-]+)/rgb\.mp4', path)
            if match:
                file = args.out/match[1]/'rgb.mp4'
                if not file.is_file():
                    return self.send(404, b'Video unavailable', 'text/plain')
                size, start, end = file.stat().st_size, 0, file.stat().st_size-1
                requested = self.headers.get('Range')
                if requested:
                    parsed = re.fullmatch(r'bytes=(\d+)-(\d*)', requested)
                    if not parsed:
                        return self.send(416, b'Invalid range', 'text/plain')
                    start, end = int(parsed[1]), min(int(parsed[2]) if parsed[2] else end, end)
                    if start > end:
                        return self.send(416, b'Invalid range', 'text/plain')
                self.send_response(206 if requested else 200)
                self.send_header('Content-Type', 'video/mp4')
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Content-Length', str(end-start+1))
                if requested:
                    self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
                self.end_headers()
                with file.open('rb') as stream:
                    stream.seek(start)
                    remaining = end-start+1
                    while remaining:
                        chunk = stream.read(min(262144, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                return
            self.proxy()

        def do_POST(self):
            if self.path.split('?')[0].startswith('/api/robot/'):
                return self.robot_api(self.path.split('?')[0])
            if args.offline and self.path.split('?')[0] in ('/api/stages/analyze','/api/library/analyze','/api/geometry/start','/api/pose/start','/api/simulation/start','/api/relations/start','/api/scene/command'):
                return self.send(409,json.dumps({'error':'Offline example: model and camera workers are not started.'}).encode(),'application/json')
            if self.path.split('?')[0].startswith('/api/deployment/'):
                return self.deployment_api(self.path.split('?')[0])
            if self.path.split('?')[0].startswith('/api/library/'):
                return self.library_api(self.path.split('?')[0])
            if self.path.split('?')[0].startswith('/api/relations/'):
                return self.relations_api(self.path.split('?')[0])
            if self.path.split('?')[0].startswith('/api/pose/'):
                return self.pose_api(self.path.split('?')[0])
            if self.path.split('?')[0].startswith('/api/scene/'):
                return self.scene_api(self.path.split('?')[0])
            if self.path.split('?')[0].startswith('/api/simulation/'):
                return self.simulation_api(self.path.split('?')[0])
            if self.path.split('?')[0].startswith('/api/geometry/'):
                return self.geometry_api(self.path.split('?')[0])
            if self.path.split('?')[0].startswith('/api/stages/'):
                return self.stage_api(self.path.split('?')[0])
            self.proxy()

        def robot_api(self,path):
            try:
                if self.command=='GET':
                    query=parse_qs(urlsplit(self.path).query)
                    if path=='/api/robot/state':value=robot_actions.state()
                    elif path=='/api/robot/job':value=robot_actions.job(query.get('id',[''])[0])
                    elif path=='/api/robot/logs':value=robot_actions.logs(query.get('id',[''])[0])
                    elif path=='/api/robot/trajectory':value=robot_actions.trajectory(query.get('id',[''])[0],query.get('after',['0'])[0])
                    else:return self.send(404,b'Not found','text/plain')
                elif self.command=='POST':
                    origin=self.headers.get('Origin')
                    if origin and urlsplit(origin).netloc!=self.headers.get('Host'):raise ValueError('다른 사이트의 실행 요청을 허용하지 않습니다.')
                    size=int(self.headers.get('Content-Length',0))
                    if not 0<size<16384:raise ValueError('잘못된 요청 크기입니다.')
                    body=json.loads(self.rfile.read(size))
                    if not isinstance(body,dict):raise ValueError('JSON 객체가 필요합니다.')
                    if path=='/api/robot/perform':value=robot_actions.perform(body)
                    elif path=='/api/robot/interrupt':value=robot_actions.interrupt(body.get('id'))
                    elif path=='/api/robot/manual-input':value=robot_actions.manual_input(body)
                    else:return self.send(404,b'Not found','text/plain')
                else:return self.send(404,b'Not found','text/plain')
                return self.send(200,json.dumps(value,ensure_ascii=False,allow_nan=False).encode(),'application/json')
            except (ValueError,OSError,TypeError,KeyError)as error:
                return self.send(409,json.dumps({'error':str(error)},ensure_ascii=False).encode(),'application/json')

        def deployment_api(self,path):
            try:
                if self.command=='GET':
                    query=parse_qs(urlsplit(self.path).query)
                    if path=='/api/deployment/catalog':value=deployment.catalog()
                    elif path=='/api/deployment/live-service':value=live_vision.state()
                    elif path=='/api/deployment/perception':value=deployment.perception.state(query.get('dir',[None])[0])
                    elif path=='/api/deployment/perception/frame':
                        image,mime=deployment.perception.frame(query.get('dir',[''])[0],query.get('format',['jpg'])[0])
                        return self.send(200,image,mime)
                    elif path=='/api/deployment/readiness':value=deployment.readiness(query.get('id',[''])[0])
                    elif path=='/api/deployment/job':value=deployment.job(query.get('policy_id',[''])[0],query.get('job_id',[''])[0])
                    else:return self.send(404,b'Not found','text/plain')
                elif self.command=='POST':
                    size=int(self.headers.get('Content-Length',0))
                    if not 0<size<16384:raise ValueError('잘못된 요청 크기입니다.')
                    body=json.loads(self.rfile.read(size))
                    if not isinstance(body,dict):raise ValueError('JSON 객체가 필요합니다.')
                    if path=='/api/deployment/register':value=deployment.register(body.get('source_dir'),body.get('name'),body.get('model_xml'))
                    elif path=='/api/deployment/live-service':
                        if args.offline:raise ValueError('오프라인 모드에서는 인식 모델을 시작하지 않습니다.')
                        value=live_vision.ensure()
                    elif path=='/api/deployment/start':value=deployment.start(body.get('policy_id'),body.get('mode'),body.get('seconds',2.),body.get('recording_id'))
                    elif path=='/api/deployment/stop':value=deployment.stop(body.get('policy_id'),body.get('job_id'))
                    elif path=='/api/deployment/plan':value=deployment.plan(body.get('policy_id'),body.get('mode','shadow'),body.get('seconds',2.),body.get('calibration'),body.get('perception_dir'),body.get('channel','can_piper'))
                    else:return self.send(404,b'Not found','text/plain')
                else:return self.send(404,b'Not found','text/plain')
                return self.send(200,json.dumps(value,ensure_ascii=False,allow_nan=False).encode(),'application/json')
            except (ValueError,OSError,TypeError,KeyError)as error:
                return self.send(409,json.dumps({'error':str(error)},ensure_ascii=False).encode(),'application/json')

        def library_api(self,path):
            try:
                if self.command=='GET':
                    if path=='/api/library/catalog':value={'assets':library.library.catalog()}
                    elif path=='/api/library/state':value=library.state(parse_qs(urlsplit(self.path).query).get('id',[''])[0])
                    else:return self.send(404,b'Not found','text/plain')
                elif self.command=='POST'and path=='/api/library/analyze':
                    size=int(self.headers.get('Content-Length',0))
                    if not 0<size<8192:raise ValueError('잘못된 요청 크기입니다.')
                    body=json.loads(self.rfile.read(size))
                    if not isinstance(body,dict):raise ValueError('JSON 객체가 필요합니다.')
                    value=library.analyze(body.get('recording_id'),body.get('selection'))
                else:return self.send(404,b'Not found','text/plain')
                return self.send(200,json.dumps(value,ensure_ascii=False,allow_nan=False).encode(),'application/json')
            except (ValueError,OSError,TypeError,KeyError)as error:
                return self.send(409,json.dumps({'error':str(error)},ensure_ascii=False).encode(),'application/json')

        def relations_api(self,path):
            try:
                if self.command=='GET':
                    recording_id=parse_qs(urlsplit(self.path).query).get('id',[''])[0]
                    if path=='/api/relations/state':value=relations.state(recording_id)
                    elif path=='/api/relations/graph':value=relations.graph(recording_id)
                    else:return self.send(404,b'Not found','text/plain')
                elif self.command=='POST'and path=='/api/relations/start':
                    size=int(self.headers.get('Content-Length',0))
                    if not 0<size<4096:raise ValueError('잘못된 요청 크기입니다.')
                    body=json.loads(self.rfile.read(size))
                    if not isinstance(body,dict):raise ValueError('JSON 객체가 필요합니다.')
                    value=relations.start(body.get('recording_id'))
                else:return self.send(404,b'Not found','text/plain')
                body=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
                if path=='/api/relations/graph'and 'gzip'in self.headers.get('Accept-Encoding',''):
                    return self.send(200,gzip.compress(body,compresslevel=3),'application/json','gzip')
                return self.send(200,body,'application/json')
            except (ValueError,OSError,TypeError)as error:
                return self.send(409,json.dumps({'error':str(error)},ensure_ascii=False).encode(),'application/json')

        def pose_api(self,path):
            try:
                if self.command=='GET'and path=='/api/pose/state':
                    value=poses.state(parse_qs(urlsplit(self.path).query).get('id',[''])[0])
                elif self.command=='POST'and path=='/api/pose/start':
                    size=int(self.headers.get('Content-Length',0))
                    if not 0<size<4096:raise ValueError('잘못된 요청 크기입니다.')
                    body=json.loads(self.rfile.read(size))
                    if not isinstance(body,dict):raise ValueError('JSON 객체가 필요합니다.')
                    value=poses.start(body.get('recording_id'))
                else:return self.send(404,b'Not found','text/plain')
                return self.send(200,json.dumps(value,ensure_ascii=False,allow_nan=False).encode(),'application/json')
            except (ValueError,OSError,TypeError)as error:
                return self.send(409,json.dumps({'error':str(error)},ensure_ascii=False).encode(),'application/json')

        def scene_api(self,path):
            try:
                if self.command=='GET' and path=='/api/scene/replay':
                    query=parse_qs(urlsplit(self.path).query)
                    value=replays.get(query.get('id',[''])[0],query.get('backend',['measured'])[0])
                elif self.command=='GET' and path=='/api/scene/state':
                    value=scenes.state(parse_qs(urlsplit(self.path).query).get('id',[''])[0])
                elif self.command=='POST' and path=='/api/scene/command':
                    size=int(self.headers.get('Content-Length',0))
                    if not 0<size<4096:raise ValueError('잘못된 요청 크기입니다.')
                    body=json.loads(self.rfile.read(size))
                    if not isinstance(body,dict):raise ValueError('JSON 객체가 필요합니다.')
                    value=scenes.command(body.get('recording_id'),body.get('action'))
                else:return self.send(404,b'Not found','text/plain')
                return self.send(200,json.dumps(value,ensure_ascii=False,allow_nan=False).encode(),'application/json')
            except (ValueError,OSError,TypeError) as error:
                return self.send(409,json.dumps({'error':str(error)},ensure_ascii=False).encode(),'application/json')

        def simulation_api(self,path):
            try:
                if self.command=='GET' and path=='/api/simulation/state':
                    value=simulation.state(parse_qs(urlsplit(self.path).query).get('id',[''])[0])
                elif self.command=='POST' and path=='/api/simulation/start':
                    size=int(self.headers.get('Content-Length',0))
                    if not 0<size<4096:raise ValueError('잘못된 요청 크기입니다.')
                    body=json.loads(self.rfile.read(size))
                    if not isinstance(body,dict):raise ValueError('JSON 객체가 필요합니다.')
                    value=simulation.start(body.get('recording_id'))
                else:return self.send(404,b'Not found','text/plain')
                return self.send(200,json.dumps(value,ensure_ascii=False,allow_nan=False).encode(),'application/json')
            except (ValueError,OSError,TypeError) as error:
                return self.send(409,json.dumps({'error':str(error)},ensure_ascii=False).encode(),'application/json')

        def geometry_api(self,path):
            try:
                if self.command=='GET' and path=='/api/geometry/state':
                    value=geometry.state(parse_qs(urlsplit(self.path).query).get('id',[''])[0])
                elif self.command=='POST' and path=='/api/geometry/start':
                    size=int(self.headers.get('Content-Length',0))
                    if not 0<size<4096:raise ValueError('잘못된 요청 크기입니다.')
                    body=json.loads(self.rfile.read(size))
                    if not isinstance(body,dict):raise ValueError('JSON 객체가 필요합니다.')
                    value=geometry.start(body.get('recording_id'))
                else:return self.send(404,b'Not found','text/plain')
                return self.send(200,json.dumps(value,ensure_ascii=False,allow_nan=False).encode(),'application/json')
            except (ValueError,OSError,TypeError) as error:
                return self.send(409,json.dumps({'error':str(error)},ensure_ascii=False).encode(),'application/json')

        def stage_api(self, path):
            def response(code, value):
                return self.send(code, json.dumps(value, ensure_ascii=False, allow_nan=False).encode(), 'application/json')
            try:
                if self.command == 'GET':
                    query = parse_qs(urlsplit(self.path).query)
                    model_id = query.get('model', [DEFAULT_MODEL])[0]
                    if path == '/api/stages/models':
                        return response(200, {'models': [model_status(key) for key in MODELS]})
                    if path == '/api/stages/recordings':
                        values = []
                        for folder in sorted(args.out.iterdir(), reverse=True):
                            if not folder.is_dir():
                                continue
                            manifest = read_json(folder/'manifest.json')
                            if manifest and manifest.get('complete'):
                                annotation = read_json(folder/'stages/draft.json', {})
                                values.append({'id': folder.name, 'task_name': manifest.get('task_name'),
                                    'duration_s': manifest.get('duration_s'),
                                    'review_status': annotation.get('review_status', 'not_analyzed')})
                        return response(200, {'recordings': values})
                    recording_id = query.get('id', [''])[0]
                    if path == '/api/stages/state':
                        return response(200, stages.state(recording_id, model_id))
                    if path == '/api/stages/export':
                        return response(200, stages.export(recording_id, model_id))
                if self.command == 'POST':
                    length = int(self.headers.get('Content-Length', 0))
                    if not 0 < length <= 262144:
                        raise ValueError('잘못된 요청 크기입니다.')
                    body = json.loads(self.rfile.read(length))
                    if not isinstance(body, dict):
                        raise ValueError('JSON 객체가 필요합니다.')
                    recording_id = body.get('recording_id')
                    model_id = body.get('model_id', DEFAULT_MODEL)
                    if path == '/api/stages/analyze':
                        return response(202, stages.start(recording_id, model_id))
                    if path in ('/api/stages/save', '/api/stages/confirm'):
                        return response(200, stages.save(recording_id, body.get('annotation', {}),
                            body.get('expected_revision'), confirm=path.endswith('/confirm'), model_id=model_id))
                return response(404, {'error': 'not found'})
            except (ValueError, OSError, TypeError, KeyError) as error:
                return response(409, {'error': str(error)})

        def proxy(self):
            if args.offline:
                if self.command!='GET':return self.send(409,json.dumps({'error':'Offline mode has no camera capture.'}).encode(),'application/json')
                if self.path.split('?')[0]!='/api/capture/state':return self.send(404,b'Offline','text/plain')
                values=[]
                for folder in sorted(args.out.glob('*'),reverse=True):
                    manifest=read_json(folder/'manifest.json',{})
                    if manifest.get('complete'):values.append(manifest)
                value={'root':str(args.out.resolve()),'camera':{'status':'offline','error':'합성 예제 · 카메라 연결 없음'},'recording':{'status':'idle'},'recordings':values,'free_gb':0,'calibration_attached':False}
                return self.send(200,json.dumps(value,ensure_ascii=False).encode(),'application/json')
            path = self.path.split('?')[0]
            allowed = {'GET': {'/api/capture/state', '/capture.jpg'},
                       'POST': {'/api/capture/start', '/api/capture/stop'}}
            if path not in allowed.get(self.command, set()):
                return self.send(404, b'Not found', 'text/plain')
            length = int(self.headers.get('Content-Length', 0))
            if not 0 <= length <= 4096:
                return self.send(413, b'Request too large', 'text/plain')
            data = self.rfile.read(length) if self.command == 'POST' else None
            request = urllib.request.Request(f'http://127.0.0.1:{args.node_port}'+self.path,
                data=data, method=self.command, headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(request, timeout=5) as response:
                    return self.send(response.status, response.read(), response.headers.get('Content-Type'))
            except urllib.error.HTTPError as error:
                return self.send(error.code, error.read(), 'application/json')
            except (urllib.error.URLError, TimeoutError):
                return self.send(503, json.dumps({'error': '촬영 노드 연결 대기 중'}).encode(), 'application/json')

        def log_message(self, *_):
            pass

    # Bind the UI first, so a busy port cannot leave a camera child behind.
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    try:
        try:
            if args.offline:raise StopIteration
            with urllib.request.urlopen(f'http://127.0.0.1:{args.node_port}/api/capture/state', timeout=1) as response:
                existing = json.load(response)
            if Path(existing['root']).resolve() != args.out.resolve():
                raise RuntimeError('기존 촬영 노드의 저장 경로가 다릅니다.')
            print('기존 촬영 노드에 연결합니다. 웹 종료 시 해당 노드를 종료하지 않습니다.', flush=True)
        except StopIteration:
            print('Offline mode: no camera process started.',flush=True)
        except urllib.error.URLError:
            child = subprocess.Popen([str(PYTHON), '-u', '-m', 'showonce_studio.showonce_capture_node',
                '--port', str(args.node_port), '--out', str(args.out), '--calib', str(args.calib)],
                env=runtime_env(), start_new_session=True)
        if not args.offline and live_vision.enabled:live_vision.ensure()
        print(f'ShowOnce Studio: http://localhost:{args.port}', flush=True)
        print('촬영 화면에서 녹화를 시작하세요. 종료: Ctrl-C (녹화 저장을 기다립니다).', flush=True)
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if child is not None and child.poll() is None and not live_vision.enabled:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=320)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == '__main__':
    main()
