"""Pose evidence gates and reference-frame isolation; no model downloads."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
from showonce_studio.showonce_pose import alignment_quality,pose_valid,directory
from showonce_studio.showonce_pose_api import PoseManager
from showonce_studio.showonce_replay import build_replay
from showonce_studio.showonce_stages import write_json


class ReferencePoseTests(unittest.TestCase):
    def test_occluding_hand_does_not_become_a_good_object_pose(self):
        render=np.ones((30,30));mask=np.ones((30,30),bool)
        self.assertTrue(alignment_quality(render,render,mask)['accepted'])
        quality=alignment_quality(render,render-.08,mask)
        self.assertFalse(quality['accepted']);self.assertEqual(quality['occluded_render_fraction'],1.)
        self.assertFalse(alignment_quality(render,np.zeros_like(render),mask)['accepted'])

    def test_transform_requires_rigid_finite_camera_pose(self):
        pose=np.eye(4);pose[2,3]=.6;self.assertTrue(pose_valid(pose))
        pose[0,0]=2;self.assertFalse(pose_valid(pose))
        pose[0,0]=float('nan');self.assertFalse(pose_valid(pose))

    def test_direct_asset_pose_not_registered_again_or_table_snapped(self):
        with tempfile.TemporaryDirectory()as temp:
            clip=Path(temp)/'20990101-000000-000009';clip.mkdir();write_json(clip/'manifest.json',{'complete':True})
            asset_root=clip/'geometry/sam3d_assets_v1';asset_root.mkdir(parents=True)
            mesh=asset_root/'mesh.ply';mesh.write_bytes(b'reference test mesh')
            asset=np.eye(4);asset[:3,3]=[.1,.2,.3]
            # Reference_frame_index intentionally does not exist in the short
            # sequence; direct FP poses never need measured-reference remapping.
            write_json(asset_root/'manifest.json',{'complete':True,'table_height_base_m':-.015,'assets':[
                {'id':'green_block','name':'cube','reference_frame_index':40,'pose_base_at_reference':np.eye(4).tolist(),
                 'simulation_initialization_dz_m':.2}]})
            tracking={'complete':True,'mesh_frame':'sam3d_asset_local','reference_assets':[
                {'id':'green_block','reference_mesh':'mesh.ply','reference_mesh_sha256':hashlib.sha256(mesh.read_bytes()).hexdigest()}],
                'frames':[{'index':0,'time_s':0.,'objects':{'green_block':{'pose_base':asset.tolist(),'status':'observed'}}}]}
            write_json(directory(clip)/'tracking.json',tracking)
            replay=build_replay(clip,'foundationpose');expected=asset.copy();expected[2,3]+=.015
            np.testing.assert_allclose(replay['frames'][0]['objects']['green_block']['pose_world'],expected)
            mesh.write_bytes(b'replaced asset')
            with self.assertRaises(ValueError):build_replay(clip,'foundationpose')

    def test_artifact_whitelist_and_symlink_escape(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);clip=root/'20990101-000000-000008';clip.mkdir();write_json(clip/'manifest.json',{'complete':True})
            folder=directory(clip);folder.mkdir(parents=True);target=folder/'tracking.json';target.write_text('{}')
            manager=PoseManager(root);self.assertEqual(manager.artifact(clip.name,'tracking.json'),target.resolve())
            for name in ('../calibration.json','debug/green_block/scene_raw.ply','overlays/../../task_definition.json'):
                with self.assertRaises(ValueError):manager.artifact(clip.name,name)
            target.unlink();target.symlink_to(clip/'manifest.json')
            with self.assertRaises(ValueError):manager.artifact(clip.name,'tracking.json')

if __name__=='__main__':unittest.main()
