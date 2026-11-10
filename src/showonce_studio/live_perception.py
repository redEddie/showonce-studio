"""Read-only live perception inspection. Never starts SAM, camera or robot I/O."""
import math
from pathlib import Path
import time
from urllib.parse import urlencode
from .showonce_stages import read_json


class LivePerception:
    def __init__(self,root):self.root=Path(root).resolve()

    def folder(self,requested):
        folder=Path(requested).expanduser().resolve()
        if self.root/'runs'not in folder.parents:raise ValueError('Studio runs 아래의 실시간 인식 폴더를 선택하세요.')
        manifest=read_json(folder/'manifest.json',{})
        if manifest.get('mode')!='live_camera_only':raise ValueError('저장 영상이 아닌 실시간 인식 노드 출력이 필요합니다.')
        return folder

    def discover(self):
        rows=[]
        for path in (self.root/'runs').glob('show100-vision-*/manifest.json'):
            try:
                folder=self.folder(path.parent)
                snapshot=read_json(folder/'tracked_poses.json',{})
                stamp=float(snapshot.get('sample_time_s',0))
                if math.isfinite(stamp):rows.append((stamp,path.stat().st_mtime,folder))
            except (ValueError,OSError,TypeError):continue
        return max(rows,key=lambda row:max(row[:2]))[2]if rows else None

    def state(self,requested=None,now=None):
        now=time.time()if now is None else now
        folder=self.folder(requested)if requested else self.discover()
        value=dict(now_s=now,connected=False,phase='not_connected',objects=[],
            preview_url='/capture.jpg',preview_kind='capture_only',image_age_s=None,
            message='실시간 인식 미연결 · 아래 영상은 촬영 카메라 원본이며 SAM 검출 결과가 아닙니다.')
        service=read_json(self.root/'runs/showonce-live-vision/service.json',{})
        try:
            alive=service['output_dir'].encode()in Path(f'/proc/{service["pid"]}/cmdline').read_bytes()
        except (KeyError,OSError):alive=False
        if alive and not requested:
            preferred=Path(service['output_dir'])
            if (preferred/'manifest.json').is_file():folder=self.folder(preferred)
            else:
                value.update(phase='starting',message='실시간 인식 시작 중 · 촬영 노드의 RGB-D 연결을 기다립니다.')
                return value
        if folder is None:return value
        manifest=read_json(folder/'manifest.json',{});snapshot=read_json(folder/'tracked_poses.json',{})
        diagnostics=read_json(folder/'diagnostics.json',{});startup=read_json(folder/'startup.json',{})
        report=read_json(folder/'report.json',{})
        try:age=now-float(snapshot['sample_time_s'])
        except (KeyError,ValueError,TypeError):age=None
        connected=age is not None and math.isfinite(age)and -.01<=age<=1. and not snapshot.get('offline_replay',True)
        value.update(connected=connected,phase=startup.get('phase','starting')if connected else 'stale',
            folder=str(folder),calibration=manifest.get('calibration',{}).get('path'),sample_age_s=age,
            error=report.get('error'),sam=diagnostics.get('sam_last_result'),
            message='실시간 인식 연결됨 · RGB에 추정 형상을 겹쳐 표시합니다.'if connected else
                '실시간 인식 응답 없음 · 마지막 검출값을 현재 관측으로 사용하지 않습니다.')
        if alive and str(folder)==service.get('output_dir')and not connected and not report.get('error'):
            value.update(phase=startup.get('phase','starting'),message='실시간 인식 준비 중 · '+startup.get('phase','RGB-D 연결 중'))
        measured={item.get('id'):item for item in snapshot.get('objects',[])}
        for oid,label in [('green_block','큐브'),('bowl','그릇')]:
            item=measured.get(oid,{})
            try:
                observed_age=now-float(item['source_time_s']);position=item['pos']
                if len(position)!=3 or not all(math.isfinite(float(v))for v in position)or not math.isfinite(observed_age):raise ValueError()
            except (KeyError,TypeError,ValueError):observed_age=None;position=None
            status=('fresh'if -.01<=observed_age<=.2 else 'delayed')if connected and observed_age is not None else 'missing'
            value['objects'].append(dict(id=oid,label=label,status=status,position_base_m=position,
                observation_age_s=observed_age,reason=diagnostics.get('object_status',{}).get(oid),
                estimator=item.get('estimator')))
        for name,mime in [('latest.jpg','image/jpeg'),('latest.png','image/png')]:
            image=folder/name
            if image.is_file():
                image_age=now-image.stat().st_mtime
                if connected and -.01<=image_age<=2.:
                    value.update(preview_url='/api/deployment/perception/frame?'+urlencode({'dir':str(folder),'format':image.suffix[1:]}),
                        preview_kind='perception_overlay',image_age_s=image_age)
                    break
        if connected and value['preview_kind']=='capture_only':
            value.update(preview_url=None,preview_kind='pending',message='인식 노드는 연결됐으며 오버레이 영상을 기다리고 있습니다.')
        return value

    def frame(self,requested,format='jpg'):
        if format not in ('jpg','png'):raise ValueError('지원하지 않는 영상 형식입니다.')
        folder=self.folder(requested);file=(folder/('latest.'+format)).resolve()
        if file.parent!=folder:raise ValueError('잘못된 미리보기 경로입니다.')
        if time.time()-file.stat().st_mtime>2. or not self.state(str(folder))['connected']:
            raise ValueError('실시간 인식 영상이 만료되었습니다.')
        return file.read_bytes(),'image/jpeg'if format=='jpg'else'image/png'
