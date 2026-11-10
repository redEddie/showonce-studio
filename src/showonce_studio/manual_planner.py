"""Fixed-TCP-position posture planning for the explicit manual unwind command.

The caller supplies model/table validity checks. This does not add self/object
collision sensing. Planning never connects to a robot or sends a target.
"""
import time
import uuid
import numpy as np
from piper_policy_runtime.control100 import ControlFault


PLAN_POSITION_TOLERANCE_M=.0002
REFERENCE_POSITION_TOLERANCE_M=.001
MEASURED_POSITION_TOLERANCE_M=.002

def position_edge_error(context,start,end,anchor):
    count=max(2,int(np.ceil(np.max(np.abs(end-start))/.005)))
    qs=np.linspace(start,end,count+1)
    return float(np.max(np.linalg.norm(context.forward_positions(qs)-anchor,axis=1)))

def unwind_plan(context, q, width, *, cancel=None):
    """Improve posture on the constant-TCP-position manifold; orientation is free.

    Project small null-space steps back onto the measured XYZ. Every segment
    is checked, including between nodes. No free-space fallback is allowed.
    """
    q=np.asarray(q,float);model=context.model;cancel=cancel or(lambda:False)
    context.check(q,q,width);anchor=model.forward(q)['tcp_position'].copy()
    span=model.upper-model.lower
    preferred=np.array([q[0],.9,-1.2,0.,.6,0.])
    weights=np.array([.15,.3,.3,3.,.3,3.])
    def objective(candidate):
        normalized=(candidate-model.lower)/span
        cost=np.sum(weights*((candidate-preferred)/span)**2)-.01*np.sum(np.log(normalized+.02)+np.log(1-normalized+.02))
        gradient=2*weights*(candidate-preferred)/span**2-.01/span*(1/(normalized+.02)-1/(1-normalized+.02))
        return float(cost),gradient
    def project(candidate):
        candidate=np.clip(candidate,model.lower+1e-8,model.upper-1e-8)
        for _ in range(12):
            error=anchor-model.forward(candidate)['tcp_position']
            if np.linalg.norm(error)<2e-6:return candidate
            jac=model.jacobian(candidate)[:3]
            correction=jac.T@np.linalg.solve(jac@jac.T+1e-10*np.eye(3),error)
            correction*=min(1.,.05/max(float(np.max(np.abs(correction))),1e-12))
            candidate=np.clip(candidate+correction,model.lower+1e-8,model.upper-1e-8)
        raise ControlFault('TCP 위치를 유지하는 관절 투영에 실패했습니다.')
    cost_before=objective(q)[0];route=[q.copy()];deadline=time.monotonic()+3.;checks=0
    for _ in range(220):
        if cancel():raise InterruptedError('위치 고정 자세 계산이 중단되었습니다.')
        if time.monotonic()>deadline:break
        cost,gradient=objective(q);jac=model.jacobian(q)[:3]
        direction=-gradient+jac.T@np.linalg.solve(jac@jac.T+1e-10*np.eye(3),jac@gradient)
        if np.max(np.abs(direction))<1e-5:break
        direction*=.02/max(float(np.max(np.abs(direction))),1e-12)
        accepted=False
        for backtrack in range(7):
            if cancel():raise InterruptedError('위치 고정 자세 계산이 중단되었습니다.')
            try:
                candidate=project(q+direction*2**(-backtrack));checks+=1
                context.check(q,candidate,width)
                if objective(candidate)[0]>=cost-1e-8:continue
                if position_edge_error(context,q,candidate,anchor)>PLAN_POSITION_TOLERANCE_M:continue
                accepted=True;break
            except (ValueError,ControlFault,np.linalg.LinAlgError):continue
        if not accepted:break
        q=candidate;route.append(q.copy())
    cost_after=objective(q)[0]
    if len(route)<2 or cost_before-cost_after<1e-4:
        raise ControlFault('TCP 위치를 고정하면서 개선할 수 있는 자세를 찾지 못했습니다. 목표를 보내지 않습니다.')
    # Shorten only edges that still preserve XYZ; do not use a free-space RRT shortcut.
    shortened=[route[0]];index=0
    while index<len(route)-1:
        if cancel():raise InterruptedError('위치 고정 자세 계산이 중단되었습니다.')
        chosen=index+1
        for candidate in range(min(index+8,len(route)-1),index,-1):
            if np.max(np.abs(route[candidate]-route[index]))>.06:continue
            checks+=1
            if position_edge_error(context,route[index],route[candidate],anchor)>PLAN_POSITION_TOLERANCE_M:continue
            try:context.check(route[index],route[candidate],width)
            except (ValueError,ControlFault):continue
            chosen=candidate;break
        shortened.append(route[chosen]);index=chosen
    route=shortened;tcp_path=[];maximum=0.
    for start,end in zip(route,route[1:]):
        context.check(start,end,width)
        error=position_edge_error(context,start,end,anchor);maximum=max(maximum,error)
        if error>PLAN_POSITION_TOLERANCE_M:raise ControlFault('TCP 위치 고정 경로의 허용 오차를 초과했습니다.')
        tcp_path.extend((context.forward_positions(np.linspace(start,end,9,endpoint=False))*1000).tolist())
    tcp_path.append((model.forward(route[-1])['tcp_position']*1000).tolist())
    from .manual_control import pose
    margin=float(np.degrees(np.min(np.minimum(route[-1]-model.lower,model.upper-route[-1]))))
    return dict(id=uuid.uuid4().hex,method='fixed-tcp-nullspace',joint_waypoints_rad=[point.tolist()for point in route],
                segments=len(route)-1,tcp_path_mm=tcp_path,start_pose=pose(model,route[0]),goal_pose=pose(model,route[-1]),
                goal_joint_margin_deg=margin,edge_checks=checks,gripper_width_m=float(width),tcp_fixed=True,
                position_anchor_m=anchor.tolist(),planned_max_tcp_error_m=maximum,
                reference_position_tolerance_m=REFERENCE_POSITION_TOLERANCE_M,
                measured_position_tolerance_m=MEASURED_POSITION_TOLERANCE_M,
                posture_cost_before=cost_before,posture_cost_after=cost_after,orientation_fixed=False,
                collision_scope='joint bounds and sampled robot/table geometry; no self/object collision',
                hardware_commands_sent=False)
