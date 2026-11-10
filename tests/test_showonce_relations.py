"""Time-aware relation evidence, uncertainty and stage independence."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from showonce_studio.showonce_relations import TemporalPredicate,VelocityWindow,threshold,pose_metrics,assemble,directory
from showonce_studio.showonce_stages import write_json
from showonce_studio.showonce_relations_api import RelationsManager


class RelationTests(unittest.TestCase):
    def test_hysteresis_and_timestamp_dwell_reject_short_pulses(self):
        state=TemporalPredicate(.1)
        self.assertEqual(state.update(0.,'false'),'unknown')
        self.assertEqual(state.update(.11,'false'),'false')
        self.assertEqual(state.update(.12,'true'),'false')
        self.assertEqual(state.update(.18,'false'),'false')
        self.assertEqual(state.update(.2,'true'),'false')
        self.assertEqual(state.update(.31,'true'),'true')
        self.assertEqual(threshold(.095,'true',.08,.11),'true')
        self.assertEqual(threshold(.095,'false',.08,.11),'false')
        self.assertEqual(state.update(.32,'unknown'),'unknown')
        self.assertEqual(state.update(.34,'false'),'unknown')

    def test_velocity_uses_real_timestamps_and_forgets_long_gaps(self):
        window=VelocityWindow()
        for t in [0.,.02,.06,.11,.15]:v=window.update(t,[.3+t*.2,-.1,0.])
        np.testing.assert_allclose(v,[.2,0,0],atol=1e-12)
        self.assertIsNone(window.update(.6,[.8,0,0]))
        self.assertIsNone(window.update(.61,None))

    def test_bowl_enclosing_box_is_not_its_cavity(self):
        vertices=np.array([[x,y,z]for x in [-.01,.01]for y in [-.01,.01]for z in [-.01,.01]])
        cavity=np.array([[-.04,-.04],[.04,-.04],[.04,.04],[-.04,.04]],np.float32)
        bounds=np.array([[-.10,-.10,-.03],[.10,.10,.03]])
        cube=np.eye(4);bowl=np.eye(4);cube[0,3]=.07
        outside=pose_metrics(cube,bowl,vertices,cavity,bounds,0)
        self.assertEqual(outside['cube_footprint_in_cavity_fraction'],0.)
        cube[0,3]=0
        self.assertEqual(pose_metrics(cube,bowl,vertices,cavity,bounds,0)['cube_footprint_in_cavity_fraction'],1.)

    def test_lost_hand_is_unknown_not_release_and_stages_do_not_drive_graph(self):
        with tempfile.TemporaryDirectory()as temp:
            clip=Path(temp)
            for name in ['geometry/foundationpose_v1/tracking.json','calibration.json','relations/scene_graph_v1/hand_tracking.json']:
                write_json(clip/name,{})
            rows=[{'index':i,'time_s':t,'geometry':None,'hand_cube_distance_m':.006 if i<3 else None,
                'hand_cube_relative_speed_m_s':.01 if i<3 else None,'cube_speed_m_s':.1,
                'cube_bowl_relative_speed_m_s':None,'selected_hand_id':1 if i<3 else None,'hands':[]}
                for i,t in enumerate([0,.12,.24,.36,.48])]
            cavity=np.array([[-.04,-.04],[.04,-.04],[.04,.04],[-.04,.04]])
            write_json(clip/'task_definition.json',{'stages':[{'start_s':0,'end_s':1}]})
            first=assemble(clip,rows,{}, {},0.,cavity,lambda *a,**k:None)
            self.assertEqual(first['frames'][2]['relations'][2]['state'],'true')
            self.assertEqual(first['frames'][3]['relations'][2]['state'],'unknown')
            self.assertFalse(any(e['relation_id']=='grasped'and e['kind']=='death'for e in first['events']))
            write_json(clip/'task_definition.json',{'stages':[{'start_s':300,'end_s':400}]})
            second=assemble(clip,rows,{}, {},0.,cavity,lambda *a,**k:None)
            self.assertEqual(first['frames'],second['frames']);self.assertEqual(first['events'],second['events'])
            self.assertFalse(first['stage_timings_used_for_inference'])

    def test_bad_recording_identifier_is_rejected(self):
        with tempfile.TemporaryDirectory()as temp:
            manager=RelationsManager(temp)
            for value in [None,'../other','20990101-000000-000001/../../secret']:
                with self.assertRaises(ValueError):manager.state(value)

if __name__=='__main__':unittest.main()
