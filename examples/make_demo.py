"""Generate a synthetic clip and zero actor: no personal video, VLM or robot."""
from pathlib import Path
import json
import shutil
import cv2
import numpy as np
from showonce_studio.config import ROOT
from showonce_studio.showonce_stages import write_json
from showonce_studio.showonce_models import DEFAULT_MODEL,result_directory
from showonce_studio.showonce_task_template import TEMPLATE_ID,STAGES,BOUNDARIES
from showonce_studio.showonce_policies import PolicyLibrary,sha
from piper_runtime.resources import MODEL_XML

ID='20990101-000000-000001'
clip=ROOT/'datasets/demonstrations'/ID
if clip.exists():raise SystemExit('Example already exists; select a new SHOWONCE_HOME to generate another copy.')
(clip/'rgb').mkdir(parents=True);(clip/'depth').mkdir()
writer=cv2.VideoWriter(str(clip/'rgb.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(640,360));assert writer.isOpened()
rows=[]
for i in range(51):
    image=np.full((360,640,3),235,np.uint8);cv2.putText(image,'SYNTHETIC DEMO - NO MODEL / ROBOT',(25,35),cv2.FONT_HERSHEY_SIMPLEX,.55,(60,70,60),1)
    cv2.ellipse(image,(440,230),(60,35),0,0,360,(100,150,200),5)
    alpha=np.clip((i-20)/15,0,1);x=int(150+alpha*290);y=int(230-70*np.sin(np.pi*alpha))
    cv2.rectangle(image,(x-22,y-22),(x+22,y+22),(70,155,70),-1);writer.write(image)
    cv2.imwrite(str(clip/f'rgb/{i:06d}.png'),image);cv2.imwrite(str(clip/f'depth/{i:06d}.png'),np.full((360,640),1000,np.uint16))
    rows.append({'index':i,'rgb':f'rgb/{i:06d}.png','depth':f'depth/{i:06d}.png','color_device_timestamp_ms':i*100,'depth_device_timestamp_ms':i*100,'host_wall_time_s':i/10})
writer.release();(clip/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n'for row in rows))
write_json(clip/'manifest.json',{'schema':'showonce_rgbd_demonstration_v1','id':ID,'task_name':'[합성 예제] 큐브를 그릇에 넣기','synthetic':True,'status':'complete','complete':True,'duration_s':5.,'frames_written':51,'video':'rgb.mp4','robot_commands_sent':False,'camera':{'intrinsics':{'width':640,'height':360,'fx':500.,'fy':500.,'cx':320.,'cy':180.}},'depth':{'meters_per_unit':.001},'note':'Synthetic illustration and constant fake depth, not a measured RGB-D demonstration.'})
annotation={'recording_id':ID,'model_id':DEFAULT_MODEL,'template_id':TEMPLATE_ID,'duration_s':5.,'revision':1,'review_status':'unreviewed','source':{'synthetic':True,'model':'fixture, no VLM inference'},'objects':[{'id':'green_block','name':'cube'},{'id':'bowl','name':'bowl'}],
 'stages':[dict(stage,id=stage['code'],start_s=float(i),end_s=float(i+1),start_frame=i*10,end_frame=(i+1)*10,description='Synthetic example interval; not a model observation.',object_ids=['green_block','bowl'])for i,stage in enumerate(STAGES)],
 'boundaries':{b['key']:{'time_s':float(i+1),'source_frame_index':(i+1)*10,'evidence':'synthetic fixture','confidence':'not measured'}for i,b in enumerate(BOUNDARIES)}}
folder=result_directory(clip,DEFAULT_MODEL)
for name in ('automatic.json','draft.json'):write_json(folder/name,annotation)
write_json(folder/'job.json',{'status':'complete','synthetic':True})
# Test-only zero policy; cannot be confused with a trained checkpoint.
source=ROOT/'artifacts/synthetic-policy';source.mkdir(parents=True)
meta=json.loads((Path(__file__).resolve().parents[1]/'tests/policy-fixture.json').read_text())
weights={'normalizer/mean':np.zeros(49,np.float32),'normalizer/std':np.ones(49,np.float32),'params/hidden_0/kernel':np.zeros((49,14),np.float32),'params/hidden_0/bias':np.zeros(14,np.float32)}
np.savez(source/'weights.npz',**weights);meta['weights_sha256']=sha(source/'weights.npz');write_json(source/'policy.json',meta)
np.savez(source/'golden.npz',observations=np.zeros((3,1,49),np.float32),resets=np.zeros((3,1),bool),actions=np.zeros((3,1,7),np.float32))
item=PolicyLibrary().register(source,'합성 예제 · 학습하지 않은 zero policy',MODEL_XML)
print(json.dumps({'recording_id':ID,'policy_id':item['id'],'synthetic':True,'hardware_commands_sent':False},ensure_ascii=False))
