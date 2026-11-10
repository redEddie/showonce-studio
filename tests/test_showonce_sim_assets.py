"""Metric RGB-D and collision-opening contracts; no model weights or robot needed."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import trimesh
from showonce_studio.showonce_asset_physics import validate
from showonce_studio.showonce_sim_assets_api import SimulationAssetsManager
from showonce_studio.showonce_stages import write_json


def fixture_scene(root,filled=False):
    objects=[]
    for oid in ['green_block','bowl']:
        folder=root/'assets'/oid;folder.mkdir(parents=True)
        if oid=='green_block':parts=[trimesh.creation.box([.025,.025,.025])];p=[.2,0,.0125]
        elif filled:parts=[trimesh.creation.box([.12,.12,.06])];p=[0,0,.03]
        else:
            parts=[];p=[0,0,.03]
            for size,pos in [([.12,.12,.005],[0,0,-.0275]),([.005,.12,.055],[-.0575,0,.0025]),
                             ([.005,.12,.055],[.0575,0,.0025]),([.11,.005,.055],[0,-.0575,.0025]),
                             ([.11,.005,.055],[0,.0575,.0025])]:
                mesh=trimesh.creation.box(size);mesh.apply_translation(pos);parts.append(mesh)
        paths=[]
        for i,part in enumerate(parts):
            path=folder/f'part_{i:03d}.obj';part.export(path);paths.append(str(path.relative_to(root)))
        mesh=trimesh.util.concatenate(parts);mesh.export(folder/'mesh.obj')
        objects.append({'id':oid,'mesh':str((folder/'mesh.obj').relative_to(root)),'extents':mesh.extents.tolist(),
            'pose_world':{'position':p,'quat_wxyz':[1,0,0,0]},'colliders':paths})
    return {'objects':objects}


class SimulationAssetTests(unittest.TestCase):
    def test_hollow_container_accepts_cube_and_settles(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);report=validate(root,fixture_scene(root))
            self.assertTrue(report['passed'],report)
            self.assertTrue(report['container_drop']['passed'])
            self.assertFalse(report['isaac_physx_executed'])

    def test_filled_container_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);report=validate(root,fixture_scene(root,filled=True))
            self.assertFalse(report['passed'])
            self.assertFalse(report['container_drop']['passed'])

    def test_simulation_artifacts_do_not_expose_weights_or_inputs(self):
        from showonce_studio.showonce_sim_assets import directory
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);clip=root/'20990101-000000-000004';clip.mkdir()
            write_json(clip/'manifest.json',{'complete':True})
            path=directory(clip)/'assets/bowl/mesh.glb';path.parent.mkdir(parents=True);path.write_bytes(b'glb')
            manager=SimulationAssetsManager(root)
            self.assertEqual(manager.artifact(clip.name,'assets/bowl/mesh.glb'),path.resolve())
            for name in ['../calibration.json','inputs/bowl/000000/pointmap_camera.npy','../../models/SAM-3D-Objects/checkpoints/ss_generator.ckpt']:
                with self.subTest(name=name),self.assertRaises(ValueError):manager.artifact(clip.name,name)


if __name__=='__main__':unittest.main()
