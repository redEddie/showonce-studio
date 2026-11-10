"""Measured geometry contracts, motion estimation and artifact isolation."""
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from showonce_studio.showonce_geometry import transform,rigid_alignment,rgbd_motion,directory
from showonce_studio.showonce_geometry_api import GeometryManager
from showonce_studio.showonce_stages import write_json


class MeasuredGeometryTests(unittest.TestCase):
    def test_rigid_alignment_recovers_metric_transform(self):
        rng=np.random.default_rng(8);source=rng.normal(size=(100,3))*.03
        angle=.35;matrix=np.eye(4);matrix[:3,:3]=[[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]
        matrix[:3,3]=[.012,-.02,.035]
        target=transform(source,matrix);estimated=rigid_alignment(source,target)
        np.testing.assert_allclose(estimated,matrix,atol=1e-10)
        np.testing.assert_allclose(transform(target,np.linalg.inv(estimated)),source,atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(estimated[:3,:3]),1.)

    def test_static_rgbd_does_not_invent_rotation(self):
        rng=np.random.default_rng(3);rgb=rng.integers(0,255,(80,80,3),dtype=np.uint8)
        depth=np.ones((80,80),np.float32);second=depth+rng.normal(0,.0003,depth.shape).astype(np.float32)
        mask=np.zeros((80,80),bool);mask[20:60,20:60]=True
        matrix,method=rgbd_motion(rgb,rgb,depth,second,mask,mask,{'cx':40,'cy':40,'fx':200,'fy':200},np.eye(4))
        self.assertIn(method,('stationary_rgbd_observation','stationary_mask_and_depth_observation'));np.testing.assert_array_equal(matrix,np.eye(4))

    def test_rgbd_pixel_motion_has_metric_scale(self):
        import cv2
        rng=np.random.default_rng(4);rgb=rng.integers(0,255,(80,80,3),dtype=np.uint8)
        after=cv2.warpAffine(rgb,np.float32([[1,0,2],[0,1,0]]),(80,80))
        mask=np.zeros((80,80),bool);mask[20:60,20:60]=True
        moved=np.zeros((80,80),bool);moved[20:60,22:62]=True
        matrix,method=rgbd_motion(rgb,after,np.ones((80,80)),np.ones((80,80)),mask,moved,
            {'cx':40,'cy':40,'fx':200,'fy':200},np.eye(4))
        self.assertEqual(method,'rgbd_optical_flow_rigid_fit')
        self.assertAlmostEqual(matrix[0,3],.01,delta=.002)
        self.assertAlmostEqual(matrix[1,3],0.,delta=.002)

    @unittest.skipUnless(importlib.util.find_spec('open3d'), 'optional geometry extra (Open3D) is not installed')
    def test_object_local_tsdf_uses_world_to_camera_extrinsic(self):
        import open3d as o3d
        intrinsic=o3d.camera.PinholeCameraIntrinsic(80,80,200.,200.,40.,40.)
        volume=o3d.pipelines.integration.UniformTSDFVolume(length=.12,resolution=40,sdf_trunc=.009,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,origin=[-.06,-.06,-.06])
        for offset in [0,2]:
            depth=np.zeros((80,80),np.float32);depth[32:48,32+offset:48+offset]=1.
            color=np.full((80,80,3),130,np.uint8)
            frame=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(color),o3d.geometry.Image(depth),
                depth_scale=1.,depth_trunc=2.,convert_rgb_to_intensity=False)
            extrinsic=np.eye(4);extrinsic[:3,3]=[offset/200,0,1]
            volume.integrate(frame,intrinsic,extrinsic)
        mesh=volume.extract_triangle_mesh()
        self.assertGreater(len(mesh.triangles),100)
        self.assertLess(np.abs(np.asarray(mesh.vertices)[:,2]).max(),.006)

    def test_artifact_whitelist_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);clip=root/'20990101-000000-000003';clip.mkdir()
            write_json(clip/'manifest.json',{'complete':True});manager=GeometryManager(root)
            valid=directory(clip)/'assets/green_block/mesh.glb';valid.parent.mkdir(parents=True);valid.write_bytes(b'mesh')
            self.assertEqual(manager.artifact(clip.name,'assets/green_block/mesh.glb'),valid.resolve())
            for name in ['../task_definition.json','assets/../../calibration.json','assets/green_block/other.json','%2e%2e/secret']:
                with self.subTest(name=name),self.assertRaises(ValueError):manager.artifact(clip.name,name)
            secret=root/'secret.json';secret.write_text('{}')
            escaped=valid.parent/'points.json';escaped.symlink_to(secret)
            with self.assertRaises(ValueError):manager.artifact(clip.name,'assets/green_block/points.json')


if __name__=='__main__':unittest.main()
