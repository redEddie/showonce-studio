"""Reference-frame mapping must preserve SAM geometry and recorded rigid motion."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from showonce_studio.showonce_replay import build_replay,ReplayManager
from showonce_studio.showonce_stages import write_json
from showonce_studio.showonce_geometry import directory as measured_directory
from showonce_studio.showonce_sim_assets import directory as asset_directory


class VideoReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.clip=self.root/'20990101-000000-000006';self.clip.mkdir()
        write_json(self.clip/'manifest.json',{'complete':True})
        self.reference=np.eye(4);self.reference[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];self.reference[:3,3]=[.3,.1,.02]
        self.asset=np.eye(4);self.asset[:3,3]=[.31,.12,.025]
        movement=np.eye(4);movement[:3,3]=[0,-.2,.1];self.moved=movement@self.reference
        self.tracks={'complete':True,'frames':[{'index':i,'time_s':i*.1,'objects':{'green_block':{'pose_base':pose,'status':'observed','fitness':.9}}}
            for i,pose in enumerate([self.reference.tolist(),self.moved.tolist(),None])]}
        write_json(measured_directory(self.clip)/'tracking.json',self.tracks)
        write_json(asset_directory(self.clip)/'manifest.json',{'complete':True,'table_height_base_m':-.015,'assets':[
            {'id':'green_block','name':'큐브','reference_frame_index':0,'pose_base_at_reference':self.asset.tolist(),'simulation_initialization_dz_m':.01}]})
    def tearDown(self):self.temp.cleanup()

    def test_reference_and_relative_motion_use_correct_matrix_order(self):
        result=build_replay(self.clip);poses=[f['objects']['green_block']['pose_world']for f in result['frames']]
        expected=self.asset.copy();expected[2,3]+=.025
        np.testing.assert_allclose(poses[0],expected,atol=1e-10)
        expected[:3,3]+=[0,-.2,.1]
        np.testing.assert_allclose(poses[1],expected,atol=1e-10)
        self.assertIsNone(poses[2]);self.assertEqual(result['frames'][2]['objects']['green_block']['pose_status'],'unresolved')
        self.assertFalse(result['policy']['physics_used'])

    def test_missing_reference_is_rejected_and_tracks_are_not_rewritten(self):
        path=measured_directory(self.clip)/'tracking.json';before=path.read_bytes()
        manager=ReplayManager(self.root);manager.get(self.clip.name);self.assertEqual(path.read_bytes(),before)
        self.tracks['frames'][0]['objects']['green_block']['pose_base']=None;write_json(path,self.tracks)
        with self.assertRaises(ValueError):manager.get(self.clip.name)


if __name__=='__main__':unittest.main()
