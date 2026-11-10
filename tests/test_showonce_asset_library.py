"""Immutable versions, current-recording coordinates and reference isolation."""
from pathlib import Path
import copy
import json
import tempfile
import unittest
import cv2
import numpy as np
import trimesh
from showonce_studio.showonce_asset_library import AssetLibrary,bundle,finalize_replay,object_specs,sha
from showonce_studio.showonce_stages import write_json,read_json
from test_showonce_sim_assets import fixture_scene


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.original=self.root/'source';self.original.mkdir()
        folder=bundle(self.original);folder.mkdir(parents=True);scene=fixture_scene(folder);assets=[]
        for obj in scene['objects']:
            role=obj['id'];path=folder/'assets'/role;model=trimesh.load(path/'mesh.obj',force='mesh')
            model.export(path/'mesh.ply');model.export(path/'mesh.glb');cv2.imwrite(str(path/'thumbnail.png'),np.zeros((8,8,3),np.uint8))
            assets.append({'id':role,'name':role,'prompt':'green cube'if role=='green_block'else 'bowl','color':'#559966',
                'mesh':f'assets/{role}/mesh.glb','mesh_ply':f'assets/{role}/mesh.ply','thumbnail':f'assets/{role}/thumbnail.png',
                'extent_m':model.extents.tolist(),'vertices':len(model.vertices),'triangles':len(model.faces),
                'units':'m','frame':'object_local','watertight':True,'collider_parts':len(obj['colliders']),
                'collision_meshes':obj['colliders'],'pose_base_at_reference':np.eye(4).tolist(),'reference_frame_index':123})
        write_json(folder/'manifest.json',{'complete':True,'assets':assets});write_json(folder/'scene.json',scene)
        self.library=AssetLibrary(self.root/'library')
        self.mapping={}
        for role in ('green_block','bowl'):
            item=self.library.register(self.original,role,role.replace('_','-'))
            self.mapping[role]={'asset_id':item['asset_id'],'version':item['version']}
        self.clip=self.root/'new-clip';self.clip.mkdir();write_json(self.clip/'calibration.json',{'T_base_cam':np.eye(4).tolist(),'table_check':{'height_m':-.015}})

    def tearDown(self):self.temp.cleanup()

    def test_version_registration_is_idempotent_and_does_not_store_source_poses(self):
        first=self.library.get('green-block','v1',verify=True)
        second=self.library.register(self.original,'green_block','green-block');self.assertEqual(first,second)
        self.assertNotIn('pose_base_at_reference',first['asset']);self.assertIsNone(first['asset']['reference_frame_index'])
        source=bundle(self.original)/'assets/green_block/mesh.ply';source.write_bytes(source.read_bytes()+b'\n')
        third=self.library.register(self.original,'green_block','green-block');self.assertEqual(third['version'],'v2')
        self.assertEqual(self.library.get('green-block','v1')['reference_mesh_sha256'],first['reference_mesh_sha256'])

    def test_role_path_and_integrity_checks(self):
        for asset_id,version in [('../other','v1'),('green-block','../../secret')]:
            with self.assertRaises(ValueError):self.library.get(asset_id,version)
        with self.assertRaises(ValueError):self.library.artifact('green-block','v1','../../secret')
        wrong=copy.deepcopy(self.mapping);wrong['green_block']=wrong['bowl']
        with self.assertRaises(ValueError):self.library.bind(self.clip,wrong)
        model=self.library.artifact('green-block','v1','mesh.ply');model.write_bytes(b'changed')
        with self.assertRaises(ValueError):self.library.bind(self.clip,self.mapping)

    def test_binding_preserves_metric_reference_and_never_copies_old_world_pose(self):
        binding=self.library.bind(self.clip,self.mapping);again=self.library.bind(self.clip,self.mapping);self.assertEqual(binding,again)
        asset=read_json(bundle(self.clip)/'manifest.json')['assets'][0]
        self.assertIsNone(asset['pose_base_at_reference'])
        self.assertEqual(sha(bundle(self.clip)/asset['mesh_ply']),binding['assets']['green_block']['reference_mesh_sha256'])
        self.assertEqual(object_specs(self.clip)[0]['prompt'],'green cube')
        self.assertIsNone(read_json(bundle(self.clip)/'scene.json')['objects'][0]['pose_world'])
        with self.assertRaises(ValueError):self.library.bind(self.original,self.mapping)

    def test_finalization_uses_new_calibration_and_current_measurements(self):
        self.library.bind(self.clip,self.mapping)
        write_json(self.clip/'manifest.json',{'camera':{'intrinsics':{'cx':2,'cy':2,'fx':100,'fy':100}},'depth':{'meters_per_unit':.001}})
        rows=[{'index':i,'color_device_timestamp_ms':1000+i*33,'depth':f'depth/{i:06d}.png'}for i in range(2)]
        (self.clip/'frames.jsonl').write_text('\n'.join(json.dumps(row)for row in rows))
        (self.clip/'depth').mkdir();cv2.imwrite(str(self.clip/'depth/000000.png'),np.full((4,4),600,np.uint16))
        poses={}
        for i,role in enumerate(('green_block','bowl')):
            pose=np.eye(4);pose[:3,3]=[.4+i*.1,.1,.2];poses[role]={'pose_base':pose.tolist()}
            path=self.clip/'geometry/rgbd_surface_v1/masks'/role;path.mkdir(parents=True);cv2.imwrite(str(path/'000000.png'),np.full((4,4),255,np.uint8))
        before=sha(self.library.folder('green-block','v1')/'manifest.json')
        finalize_replay(self.clip,{'frames':[{'index':0,'time_s':0,'objects':poses}]})
        scene=read_json(bundle(self.clip)/'scene.json');np.testing.assert_allclose(scene['objects'][0]['pose_world']['position'],[.4,.1,.215])
        manifest=read_json(bundle(self.clip)/'manifest.json');self.assertTrue(manifest['replay_ready']);self.assertFalse(manifest['simulation_ready'])
        points=read_json(bundle(self.clip)/manifest['assets'][0]['point_cloud']);self.assertEqual(points['source_recording_id'],self.clip.name)
        self.assertEqual(before,sha(self.library.folder('green-block','v1')/'manifest.json'))

if __name__=='__main__':unittest.main()
