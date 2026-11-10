"""The web scene returns actual simulator poses and respects play/pause/reset."""
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from showonce_studio.showonce_scene_node import SceneWorld
from showonce_studio.showonce_sim_assets import directory
from showonce_studio.showonce_stages import write_json
from showonce_studio.showonce_asset_physics import scene_xml
from test_showonce_sim_assets import fixture_scene


class SceneRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.clip=self.root/'20990101-000000-000005';self.clip.mkdir()
        write_json(self.clip/'manifest.json',{'complete':True})
        folder=directory(self.clip);folder.mkdir(parents=True)
        scene=fixture_scene(folder)
        write_json(folder/'manifest.json',{'simulation_ready':True,'assets':[{'id':obj['id']}for obj in scene['objects']]})
        ET.ElementTree(scene_xml(folder,scene)).write(folder/'scene.xml',encoding='unicode')
        ET.ElementTree(scene_xml(folder,scene,drop_test=True)).write(folder/'container-check.xml',encoding='unicode')
        self.world=SceneWorld(self.root)
    def tearDown(self):self.temp.cleanup()

    def test_actual_gravity_pause_and_reset(self):
        value=self.world.command(self.clip.name,'load')
        self.assertFalse(value['running']);self.assertEqual(value['time_s'],0.)
        self.world.advance(.1);self.assertEqual(self.world.state()['time_s'],0.)
        value=self.world.command(self.clip.name,'drop');initial=value['objects']['green_block']['position'][2]
        self.assertTrue(value['running']);self.assertEqual(self.world.model.nv,6)
        self.world.advance(.1);value=self.world.state()
        self.assertAlmostEqual(value['time_s'],.1,places=8)
        self.assertLess(value['objects']['green_block']['position'][2],initial-.02)
        self.world.command(self.clip.name,'pause');self.world.advance(.1)
        self.assertAlmostEqual(self.world.state()['time_s'],.1,places=8)
        value=self.world.command(self.clip.name,'reset')
        self.assertFalse(value['running']);self.assertEqual(value['time_s'],0.);self.assertEqual(value['scenario'],'initial')
        self.assertEqual(self.world.model.nv,12)
        np.testing.assert_allclose(value['objects']['green_block']['quat_wxyz'],[1,0,0,0],atol=1e-10)

    def test_invalid_actions_and_unvalidated_assets_are_rejected(self):
        with self.assertRaises(ValueError):self.world.command(self.clip.name,'robot_enable')
        with self.assertRaises(ValueError):self.world.command(self.clip.name,'play')
        write_json(directory(self.clip)/'manifest.json',{'simulation_ready':False})
        with self.assertRaises(ValueError):self.world.command(self.clip.name,'load')

    def test_contacts_follow_physics_instead_of_the_video_clock(self):
        initial=self.world.command(self.clip.name,'drop')
        self.assertFalse(any({p['a'],p['b']}=={'green_block','bowl'}for p in initial['contact_pairs']))
        for _ in range(30):self.world.advance(.1)
        settled=self.world.state()
        self.assertTrue(any({p['a'],p['b']}=={'green_block','bowl'}and p['count']>0 for p in settled['contact_pairs']))
        self.assertTrue(all(p['a']in ('green_block','bowl','table')and p['b']in ('green_block','bowl','table')for p in settled['contact_pairs']))
        with self.assertRaises(ValueError):self.world.command('../private','load')


if __name__=='__main__':unittest.main()
