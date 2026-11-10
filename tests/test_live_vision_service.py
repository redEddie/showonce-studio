import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from showonce_studio.showonce_stages import write_json
from showonce_studio.live_vision_service import LiveVisionService


def test_existing_worker_is_reused_and_wrong_pid_birth_is_not_owned(tmp_path):
    with patch('showonce_studio.live_vision_service.ROOT',tmp_path):
        manager=LiveVisionService(tmp_path/'calibration.json',8768)
    output=tmp_path/'runs/show100-vision-fixture'
    process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(15)',str(output)])
    try:
        birth=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
        record=dict(pid=process.pid,process_birth=birth,output_dir=str(output))
        write_json(manager.home/'service.json',record)
        assert manager.state()['alive']
        assert manager.state()['implementation']=='legacy_external_inspector'
        assert manager.state()['external_source_required']
        with patch('showonce_studio.live_vision_service.subprocess.Popen')as spawn:
            assert manager.ensure()['pid']==process.pid
            spawn.assert_not_called()
        write_json(manager.home/'service.json',dict(record,process_birth='not-this-process'))
        assert not manager.state()['alive']
        assert process.poll()is None
    finally:
        process.terminate();process.wait(timeout=3)
