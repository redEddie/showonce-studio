"""Weights-free robot and controller settings shared by Home and the tutorial."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from piper_policy_runtime.control100 import ControlFault
from piper_policy_runtime.kinematics import PiperKinematics
from piper_runtime.resources import MODEL_XML

DEFAULT_CONFIG = Path(__file__).parent/'data/posture-a.json'


def motion_adapter(policy_id=None):
    env = json.loads(DEFAULT_CONFIG.read_text())
    if policy_id:
        # Only robot configuration is required. Never open actor weights/golden vectors.
        from .showonce_policies import PolicyLibrary
        library = PolicyLibrary()
        manifest = library.get(policy_id, verify=False)
        folder = library.folder(policy_id)
        for name in ('policy.json', 'piper.xml'):
            if hashlib.sha256((folder/name).read_bytes()).hexdigest() != manifest['files'][name]:
                raise ControlFault('Policy robot configuration has changed: '+name)
        meta = json.loads((folder/'policy.json').read_text())
        if hashlib.sha256(MODEL_XML.read_bytes()).hexdigest() != meta['kinematics']['piper_xml_sha256']:
            raise ControlFault('Policy robot model differs from packaged posture model')
        if not np.allclose(meta['kinematics']['tcp_local_position_m'], [0,0,.1385], atol=1e-9):
            raise ControlFault('Posture-A requires the 138.5mm TCP')
        env.update({key:meta['environment'][key] for key in env if key in meta['environment']})
    model = PiperKinematics(MODEL_XML)
    model.tcp_local_position = np.array([0.,0.,.1385])
    target = np.asarray(env['nominal_qpos'], float)
    if target.shape != (6,) or not np.isfinite(target).all() or np.any(target<model.lower) or np.any(target>model.upper):
        raise ControlFault('Invalid posture-A joint target')
    return SimpleNamespace(model=model,environment=env)
