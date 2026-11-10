"""Produce a markerless camera calibration candidate without moving the robot.

The D435i must be free (stop the capture node first). Existing calibration is
read only for its serial. Results are never activated automatically.
"""
import argparse
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from ..config import ROOT, default_calibration, runtime_env, worker_python

DATA = Path(__file__).parent/'data'


def check_feedback(feedback):
    if not feedback.get('telemetry_fresh_at_frame'):
        raise ValueError('촬영 시점의 관절·속도·그리퍼 피드백이 최신이 아닙니다')
    for key, length in [('q_rad', 6), ('qvel_rad_s', 6)]:
        values = feedback.get(key)
        if not isinstance(values, list) or len(values) != length or any(
                not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError('불완전하거나 유효하지 않은 관절 피드백: '+key)
    if any(abs(v) > .02 for v in feedback['qvel_rad_s']):
        raise ValueError('팔이 정지하지 않았습니다 (관절 속도 >0.02rad/s)')
    width = feedback.get('gripper_full_width_m')
    if not isinstance(width, (int, float)) or not math.isfinite(width) or not 0 <= width <= .101:
        raise ValueError('유효한 실측 그리퍼 폭이 필요합니다')


def check_candidate(cal, serial, mount):
    table = cal.get('table_check') or {}
    def number(value):
        return isinstance(value, (int, float)) and math.isfinite(value)
    checks = dict(trusted=cal.get('trusted') is True,
                  serial_matches=bool(serial) and str(cal.get('serial')) == str(serial),
                  table_height_difference_mm=round(1000*abs(table['height_m']-mount), 1) if number(table.get('height_m')) else None,
                  table_tilt_deg=table.get('tilt_deg') if number(table.get('tilt_deg')) else None,
                  table_rms_mm=table.get('plane_rms_mm') if number(table.get('plane_rms_mm')) else None,
                  table_points=table.get('n_points') if number(table.get('n_points')) else None)
    passed = checks['trusted'] and checks['serial_matches']
    for key, low, high in [('table_height_difference_mm',0,10), ('table_tilt_deg',0,5),
                           ('table_rms_mm',0,5), ('table_points',1000,float('inf'))]:
        value = checks[key]
        passed = passed and number(value) and low <= value <= high
    return checks, bool(passed)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT/'runs'/('camera-calib-'+dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')))
    ap.add_argument('--serial', default=None)
    ap.add_argument('--can', default='can_piper')
    ap.add_argument('--calib', type=Path, default=default_calibration(), help='Existing calibration, read only for camera serial')
    ap.add_argument('--robot-box', type=int, nargs=4, default=[0,0,848,480])
    ap.add_argument('--mount', type=Path, default=DATA/'robot_mount.json', help='Measured table height relative to base; default -15mm')
    ap.add_argument('--src', type=Path, help='Use an existing RGB-D snapshot instead of opening camera/CAN')
    ap.add_argument('--feedback', type=Path, help='Required with --src; feedback captured with that snapshot')
    args = ap.parse_args(argv)
    if args.out.exists():ap.error('Output directory already exists')
    if bool(args.src) != bool(args.feedback):ap.error('--src and --feedback must be supplied together')
    old = json.loads(args.calib.read_text()) if args.calib.is_file() else {}
    serial = args.serial or old.get('serial')
    if args.src:
        meta = json.loads((args.src/'meta.json').read_text())
        serial = args.serial or meta.get('serial')
    if not serial:ap.error('Camera serial is missing; pass --serial')
    x0,y0,x1,y1 = args.robot_box
    if not (0<=x0<x1<=848 and 0<=y0<y1<=480):ap.error('robot-box must be within 848x480')
    mount = json.loads(args.mount.read_text())['table_z_in_base_m']
    if not isinstance(mount,(int,float)) or not math.isfinite(mount):ap.error('Invalid measured table height')
    args.out.mkdir(parents=True)
    env = runtime_env();env.pop('PYTHONPATH',None);env['PYTHONSAFEPATH']='1'
    # Child SAM and registration environments have their own installed packages.
    lock = None
    summary = dict(capture=str(args.out/'capture') if not args.src else str(args.src), camera_serial=serial,
                   hardware_commands_sent=False, active_calibration_replaced=False, offline_snapshot=bool(args.src))
    try:
        if not args.src:
            if os.environ.get('SHOW100_CALIBRATION_LOCK_HELD') != '1':
                lock = (ROOT/'runs/.show100-vision-camera.lock').open('a')
                fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            print('D435i 촬영·수신 전용 관절 피드백 수집 중. 로봇과 카메라를 움직이지 마세요.',flush=True)
            command = [str(worker_python('capture')), '-m','showonce_studio.calibration.capture',
                       '--output',str(args.out/'capture'),'--serial',str(serial),'--can',args.can,
                       '--width','848','--height','480','--frames','5']
            subprocess.run(command,env=env,check=True)
            args.src = args.out/'capture/camera';args.feedback = args.out/'capture/robot_feedback.json'
        check_feedback(json.loads(args.feedback.read_text()))
        print('SAM 로봇 마스크 → 로봇 렌더링 정합 → 베이스 마스크 정제 → 재정합·테이블 검증 중...',flush=True)
        command = [sys.executable,'-m','showonce_studio.calibration.snapshot',
                   '--src',str(args.src),'--feedback',str(args.feedback),
                   '--out',str(args.out/'registration'),'--mount',str(args.mount),
                   '--robot-box',*map(str,args.robot_box)]
        result = subprocess.run(command,env=env,capture_output=True,text=True)
        (args.out/'registration.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('보정 계산 실패. registration.log 확인: '+result.stderr[-1000:])
        print(result.stdout,flush=True)
        candidate = args.out/'registration/T_base_cam_robot.json'
        checks,passed = check_candidate(json.loads(candidate.read_text()),serial,mount)
        summary.update(candidate=str(candidate),check_png=str(candidate.with_name('check.png')),
                       checks=checks,candidate_passed=passed)
        print('보정 후보:',candidate)
        print('검증 영상:',summary['check_png'])
        print('자동 검사:','통과' if passed else '확인 필요',checks)
        print('현재 보정은 유지됩니다. 후보를 검토한 뒤 --calib 경로로 선택하세요.')
    except Exception as error:
        summary['error']=f'{type(error).__name__}: {error}'
        print(summary['error'],file=sys.stderr)
    finally:
        (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        if lock is not None:lock.close()
    return 0 if summary.get('candidate_passed') and not summary.get('error') else 1


if __name__=='__main__':raise SystemExit(main())
