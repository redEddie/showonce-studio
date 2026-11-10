"""Deployment grasp evidence layered over the locked runtime scene memory.

The actor, observation layout, controller, and registered policy stay unchanged.
A recent approach can establish the spatial evidence *before* the aperture
plateau: checking proximity only after closing missed an already lifted cube.
This remains an inference, not measured contact or a verified task success.
"""
from collections import deque
import numpy as np
from piper_policy_runtime.scene_state import SceneState
from piper_policy_runtime.show100 import vector


class DeploymentSceneState(SceneState):
    schema='showonce_scene_state_v2'
    approach_memory_s=.35
    closing_timeout_s=1.
    max_closing_travel_m=.12
    plateau_duration_s=.1
    plateau_window_s=.25
    plateau_range_m=.002
    width_band_m=.006

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.approaches=deque(maxlen=16)
        self.plateau=deque(maxlen=8)
        self.close_started=None
        self.close_anchor=None
        self.grasp_evidence=None

    def start_episode(self,now):
        super().start_episode(now)
        self.approaches.clear();self.plateau.clear()
        self.close_started=self.close_anchor=self.grasp_evidence=None

    def _remember_closing(self,robot,now,goal):
        # Leave error reporting and all validity checks to the locked estimator.
        # Do not update evidence on feedback it would reject.
        if (not self.ready or self.heartbeat is None or now-self.heartbeat>self.heartbeat_timeout
                or not -.01<=now-float(robot['sample_time_s'])<=.1):return
        if self.phase!='stationary':
            self.approaches.clear();self.plateau.clear();self.close_anchor=None
            return
        tcp=vector(self.model.forward(robot['q_rad'])['tcp_position'],3,'TCP')
        width=float(robot['gripper_full_width_m']);side=2*self.half
        if not np.isfinite([now,width,goal]).all():return
        while self.approaches and now-self.approaches[0]['time_s']>self.approach_memory_s:
            self.approaches.popleft()
        block=self.rest['green_block']
        near=(np.linalg.norm(tcp-block)<.04 or
            (np.linalg.norm(tcp[:2]-block[:2])<.05 and abs(tcp[2]-block[2])<.06))
        if near and width>side+self.width_band_m:
            self.approaches.append(dict(time_s=now,tcp=tcp.copy(),block=block.copy()))
        closing=goal<side-self.width_band_m
        if not closing:
            self.close_started=None;self.close_anchor=None;self.plateau.clear()
            return
        if self.close_started is None:
            self.close_started=now
            self.grasp_evidence=None
        if now-self.close_started>self.closing_timeout_s:
            self.close_anchor=None;self.plateau.clear();return
        if self.close_anchor is None and self.approaches:
            self.close_anchor=self.approaches[-1]
        if (self.close_anchor is None
                or np.linalg.norm(tcp-self.close_anchor['tcp'])>self.max_closing_travel_m
                or abs(width-side)>self.width_band_m):
            self.plateau.clear();return
        # A paused slew reference is not contact. Require a previously sent
        # reference that is still requesting at least6mm more closure.
        reference=robot.get('gripper_reference_width_m')
        if reference is None or not np.isfinite(reference) or reference>width-.006:
            self.plateau.clear();return
        if not self.plateau or now>self.plateau[-1][0]+1e-6:
            self.plateau.append((now,width))
        while self.plateau and now-self.plateau[0][0]>self.plateau_window_s:
            self.plateau.popleft()
        if len(self.plateau)<2:return
        duration=now-self.plateau[0][0]
        spread=float(np.ptp([w for _,w in self.plateau]))
        if duration+1e-6<self.plateau_duration_s or spread>self.plateau_range_m:return
        self.grasp_evidence=dict(reason='width_plateau_after_recent_approach',inferred=True,
            confirmed_time_s=now,approach_time_s=self.close_anchor['time_s'],
            approach_tcp_base_m=self.close_anchor['tcp'].tolist(),
            approach_cube_base_m=self.close_anchor['block'].tolist(),
            width_m=width,reference_width_m=float(reference),
            plateau_duration_s=duration,plateau_range_m=spread,
            current_tcp_to_stored_cube_m=float(np.linalg.norm(tcp-block)))
        self.phase,self.transition='held_tcp',now
        self.grasp_since=None
        self.counts['grasp_from_approach_memory']+=1

    def estimate(self,robot,now,gripper_goal):
        before=self.phase
        self._remember_closing(robot,now,gripper_goal)
        result=super().estimate(robot,now,gripper_goal)
        diagnostic=result['state_estimation']
        if before=='stationary' and self.phase=='held_tcp' and self.grasp_evidence is None:
            self.grasp_evidence=dict(reason='current_proximity_and_width_stall',inferred=True,
                confirmed_time_s=now,width_m=float(robot['gripper_full_width_m']))
        diagnostic.update(estimator_revision=self.schema,grasp_evidence=self.grasp_evidence,
            closing_attempt_age_s=None if self.close_started is None else max(0.,now-self.close_started),
            approach_memory_eligible=self.close_anchor is not None,
            plateau_samples=len(self.plateau))
        return result
