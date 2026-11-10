"""Camera/TCP comparison using passive CAN only; never construct the SDK.

The marker is encoder-derived FK projected with the existing calibration.
It must be compared with the *visible physical gripper*: projecting the cube
with the same calibration cannot independently validate absolute accuracy.
"""
import argparse
import io
import json
from pathlib import Path
import time
import urllib.request

import numpy as np


def inspect_frame(packet, feedback, calibration, model, now):
    """Check identity/timing and compute FK without any device access."""
    camera=packet['camera'];stamp=packet['stamps']
    if (packet.get('schema')!='showonce_live_rgbd_v1' or packet.get('offline_replay') is not False
            or packet.get('aligned_to')!='color' or not calibration.get('trusted')
            or str(camera['serial'])!=str(calibration.get('serial'))):
        raise ValueError('현재 보정과 일치하는 실시간 RGB-D 카메라가 필요합니다.')
    frame_time=float(stamp['host_wall_time_s']);frame_mono=float(stamp['host_monotonic_s'])
    if not np.isfinite([frame_time,frame_mono,now]).all() or not -.01<=now-frame_time<=.5:
        raise ValueError('카메라 프레임이 만료되었습니다.')
    if feedback.get('source')!='passive_socketcan_receive_only' or not feedback.get('telemetry_fresh_at_frame'):
        raise ValueError('영상 시각에 대응하는 최신 CAN 피드백을 기다립니다.')
    components=feedback['feedback_components']
    joint_ids=(0x2A5,0x2A6,0x2A7)
    monotonic=[components[f'0x{i:03X}']['host_monotonic_ns']/1e9 for i in joint_ids]
    if not all(0<=frame_mono-t<=.05 for t in monotonic) or max(monotonic)-min(monotonic)>.025:
        raise ValueError('영상과 관절 피드백의 시각 차이가 큽니다.')
    q=np.asarray(feedback['q_rad'],float);width=float(feedback['gripper_full_width_m'])
    if q.shape!=(6,) or not np.isfinite(q).all() or not np.isfinite(width):
        raise ValueError('관절각 또는 그리퍼 피드백이 유효하지 않습니다.')
    fk=model.forward(q)
    return dict(frame_time_s=frame_time,frame_sequence=int(packet['sequence']),
        stream_id=packet['stream_id'],q_rad=q.tolist(),gripper_width_m=width,
        joint_age_at_frame_ms=[1000*(frame_mono-t) for t in monotonic],
        tcp_position_base_m=np.asarray(fk['tcp_position']).tolist(),
        tcp_rotation_wxyz=np.asarray(fk['link6_quaternion_wxyz']).tolist(),
        hardware_commands_sent=False,independent_tcp_measurement=False)


def project(points, calibration, intrinsics):
    points=np.asarray(points,float)
    transform=np.asarray(calibration['T_base_cam'],float)
    if (points.ndim!=2 or points.shape[1]!=3 or transform.shape!=(4,4)
            or not np.isfinite(points).all() or not np.isfinite(transform).all()
            or not np.allclose(transform[3],[0,0,0,1])):
        raise ValueError('잘못된 투영 좌표입니다.')
    rotation=transform[:3,:3]
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-4) or np.linalg.det(rotation)<0:
        raise ValueError('보정 회전 행렬이 유효하지 않습니다.')
    camera=(np.linalg.inv(transform)@np.c_[points,np.ones(len(points))].T).T[:,:3]
    values=np.array([intrinsics[k] for k in ('fx','fy','cx','cy')],float)
    if not np.isfinite(values).all() or np.any(values[:2]<=0):raise ValueError('잘못된 카메라 내부 파라미터입니다.')
    if any(abs(float(c))>1e-9 for c in intrinsics.get('distortion_coefficients',[])):
        raise ValueError('이 진단은 왜곡이 보정된 영상이 필요합니다.')
    pixels=np.full((len(camera),2),np.nan)
    valid=camera[:,2]>.01
    pixels[valid]=camera[valid,:2]/camera[valid,2:3]*values[:2]+values[2:]
    return pixels


def overlay(bgr, row, calibration, intrinsics):
    import cv2
    from piper_policy_runtime.kinematics import _matrix
    tcp=np.asarray(row['tcp_position_base_m'])
    rotation=_matrix(row['tcp_rotation_wxyz'])
    points=np.vstack([tcp,*[tcp+.025*rotation[:,i] for i in range(3)]])
    pixels=project(points,calibration,intrinsics)
    image=bgr.copy();h,w=image.shape[:2]
    def visible(pixel):return np.isfinite(pixel).all() and 0<=pixel[0]<w and 0<=pixel[1]<h
    if visible(pixels[0]):
        p=tuple(np.rint(pixels[0]).astype(int))
        for i,color in enumerate(((70,80,245),(70,220,70),(245,130,70))):
            if visible(pixels[i+1]):cv2.line(image,p,tuple(np.rint(pixels[i+1]).astype(int)),color,2)
        cv2.circle(image,p,9,(255,255,0),2)
        cv2.drawMarker(image,p,(255,255,0),cv2.MARKER_CROSS,24,2)
        cv2.putText(image,'FK TCP 138.5mm', (min(max(5,p[0]+12),w-210),max(45,p[1]-10)),
            cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,0),1,cv2.LINE_AA)
        row['tcp_pixel']=pixels[0].tolist()
    else:row['tcp_pixel']=None
    text='PASSIVE CAN / NO COMMANDS | compare marker with physical gripper'
    cv2.putText(image,text,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1,cv2.LINE_AA)
    return image


def main():
    from .config import ROOT,default_calibration
    from .showonce_policies import PolicyLibrary
    from .showonce_stages import read_json
    from piper_policy_runtime.kinematics import PiperKinematics
    from piper_runtime.transports.passive_can import PassiveCAN,tx_packets
    import cv2
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy-id',required=True)
    p.add_argument('--calib',type=Path,default=default_calibration())
    p.add_argument('--channel',default='can_piper')
    p.add_argument('--capture-url',default='http://127.0.0.1:8768')
    p.add_argument('--seconds',type=float,default=3.)
    p.add_argument('--view',action='store_true')
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    if not np.isfinite(args.seconds) or not .1<=args.seconds<=120:p.error('seconds: 0.1..120')
    library=PolicyLibrary();library.get(args.policy_id);folder=library.folder(args.policy_id)
    meta=read_json(folder/'policy.json');calib=read_json(args.calib,{})
    if not isinstance(calib,dict) or not calib.get('trusted'):
        p.error('--calib에 현재 실시간 인식과 같은 신뢰된 보정 파일을 지정하세요.')
    model=PiperKinematics(folder/'piper.xml')
    model.tcp_local_position=np.asarray(meta['kinematics']['tcp_local_position_m'],float)
    if not np.allclose(model.tcp_local_position,[0,0,.1385]):p.error('TCP 138.5mm 정책을 선택하세요.')
    out=args.out or ROOT/'runs/showonce-tcp-check'/str(time.time_ns());out.mkdir(parents=True,exist_ok=False)
    (out/'calibration.json').write_text(json.dumps(calib,ensure_ascii=False,indent=2)+'\n')
    receiver=PassiveCAN(args.channel);before=tx_packets(args.channel);count=0;last=-1;error=None
    report=dict(hardware_commands_sent=False,can_source='passive receive only; SDK not imported',
        policy_id=args.policy_id,calibration=str(args.calib),tcp_local_position_m=model.tcp_local_position.tolist(),
        scope='Visual FK marker, not independent physical TCP measurement. '
              'Compare several stationary poses before changing offsets. '
              'The interface TX counter also includes other applications.')
    try:
        receiver.start()
        if receiver.errors:raise ValueError('; '.join(receiver.errors))
        deadline=time.monotonic()+args.seconds;warmup=time.monotonic()+1.
        with (out/'samples.jsonl').open('w') as log:
            while time.monotonic()<deadline:
                with urllib.request.urlopen(args.capture_url.rstrip('/')+'/api/capture/rgbd?after='+str(last),timeout=2.) as response:
                    with np.load(io.BytesIO(response.read(4*1024*1024)),allow_pickle=False) as packet:
                        bgr=packet['bgr'].copy();metadata=json.loads(packet['metadata'].tobytes())
                if int(metadata['sequence'])==last:continue
                last=int(metadata['sequence']);intr=metadata['camera']['intrinsics']
                if bgr.shape!=(int(intr['height']),int(intr['width']),3) or bgr.dtype!=np.uint8:
                    raise ValueError('RGB 영상 크기/자료형이 내부 파라미터와 다릅니다.')
                if report.get('camera_stream_id',metadata['stream_id'])!=metadata['stream_id']:
                    raise ValueError('카메라가 재시작되었습니다. 진단을 다시 실행하세요.')
                report['camera_stream_id']=metadata['stream_id'];report['camera']=metadata['camera']
                feedback=receiver.feedback(round(metadata['stamps']['host_monotonic_s']*1e9),max_age_s=.05)
                try:row=inspect_frame(metadata,feedback,calib,model,time.time())
                except ValueError:
                    if count==0 and time.monotonic()<warmup:continue
                    raise
                annotated=overlay(bgr,row,calib,intr)
                if not cv2.imwrite(str(out/f'{count:05d}.jpg'),annotated):raise OSError('진단 영상 저장 실패')
                if count==0:cv2.imwrite(str(out/'first-raw.jpg'),bgr)
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();count+=1
                if args.view:
                    cv2.imshow('ShowOnce TCP check - passive CAN',annotated)
                    if cv2.waitKey(1)&0xff in (27,ord('q')):break
                time.sleep(.1)
        if not count:raise ValueError('시각이 맞는 영상·관절 피드백을 얻지 못했습니다.')
    except KeyboardInterrupt:report['interrupted']=True
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    finally:
        receiver.stop()
        if args.view:cv2.destroyAllWindows()
        after=tx_packets(args.channel)
        report.update(success=bool(count and not error),frames=count,error=error,
            interface_tx_packet_delta=None if before is None or after is None else after-before)
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(str(out))
    print('관찰 전용 · 로봇 명령 없음 · 영상',count,'장')
    if error:raise SystemExit(error)


if __name__=='__main__':main()
