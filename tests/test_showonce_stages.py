"""Temporal partition, provenance and review-state regression tests; no GPU/CAN."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from showonce_studio.showonce_stages import StageManager, validate_annotation, write_json, read_json, recording_path
from showonce_studio.showonce_models import result_directory, DEFAULT_MODEL
from showonce_studio.showonce_task_template import (TEMPLATE_ID, BOUNDARIES, assessment_from_output,
                                    annotation_from_assessment, validate_template, boundary_prompt, protocol_id)


class StageReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.id = '20990101-000000-000001'
        self.path = self.root/self.id
        self.path.mkdir()
        self.directory = result_directory(self.path, DEFAULT_MODEL)
        self.times = [i/10 for i in range(11)]
        write_json(self.path/'manifest.json', {'complete': True, 'duration_s': 1., 'task_name': 'test'})
        (self.path/'frames.jsonl').write_text(''.join(json.dumps({'color_device_timestamp_ms': t*1000})+'\n' for t in self.times))
        self.value = {'recording_id': self.id, 'revision': 1, 'review_status': 'unreviewed',
            'source': {'model': 'fixture'}, 'duration_s': 1.,
            'objects': [{'id': 'cube', 'name': 'cube'}], 'stages': [
                {'id': 'a', 'label': 'move', 'description': 'cube moves', 'goal': 'cube raised',
                 'object_ids': ['cube'], 'start_s': 0., 'end_s': .5},
                {'id': 'b', 'label': 'place', 'description': 'cube rests', 'goal': 'cube placed',
                 'object_ids': ['cube'], 'start_s': .5, 'end_s': 1.}]}
        write_json(self.directory/'automatic.json', self.value)
        write_json(self.directory/'draft.json', self.value)
        self.manager = StageManager(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_edit_confirm_and_supersede_preserves_original(self):
        edited = deepcopy(self.value)
        edited['stages'][0]['label'] = '사용자가 수정한 단계'
        edited['source'] = {'model': 'forged browser value'}
        saved = self.manager.save(self.id, edited, 1)['annotation']
        self.assertEqual(saved['revision'], 2)
        self.assertEqual(saved['review_status'], 'draft')
        self.assertEqual(saved['source']['model'], 'fixture')
        self.assertFalse((self.path/'task_definition.json').exists())
        self.manager.save(self.id, saved, 2, confirm=True)
        task = read_json(self.path/'task_definition.json')
        self.assertEqual(task['review_status'], 'confirmed')
        self.assertFalse(task['training_ready'])
        self.assertEqual(read_json(self.directory/'automatic.json'), self.value)
        self.manager.save(self.id, saved, 3)
        self.assertEqual(read_json(self.path/'task_definition.json')['superseded_by_draft_revision'], 4)
        self.assertEqual(len(list((self.directory/'history').glob('review-*.json'))), 3)

    def test_conflicting_browser_revision_is_rejected(self):
        self.manager.save(self.id, self.value, 1)
        with self.assertRaisesRegex(ValueError, '다른 화면'):
            self.manager.save(self.id, self.value, 1)

    def test_invalid_partition_is_rejected(self):
        for change in ('overlap', 'gap', 'nan', 'outside', 'unknown_object', 'duplicate_id'):
            value = deepcopy(self.value)
            if change == 'overlap': value['stages'][1]['start_s'] = .3
            if change == 'gap': value['stages'][1]['start_s'] = .7
            if change == 'nan': value['stages'][0]['end_s'] = float('nan')
            if change == 'outside': value['stages'][1]['end_s'] = 2.
            if change == 'unknown_object': value['stages'][0]['object_ids'] = ['missing']
            if change == 'duplicate_id': value['stages'][1]['id'] = 'a'
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_annotation(value, 1., self.times)

    def test_cached_analysis_does_not_start_model(self):
        result = self.manager.start(self.id)
        self.assertIsNone(self.manager.process)
        self.assertEqual(result['automatic']['source']['model'], 'fixture')

    def test_recording_traversal_and_incomplete_recording_rejected(self):
        for value in ('../manifest.json', '/tmp/whatever', None):
            with self.assertRaises(ValueError): recording_path(self.root, value)
        write_json(self.path/'manifest.json', {'complete': False})
        with self.assertRaises(ValueError): recording_path(self.root, self.id)

    def test_model_results_reviews_and_exports_are_isolated(self):
        qwen_bytes = (self.directory/'draft.json').read_bytes()
        korean = deepcopy(self.value)
        korean.update(model_id='kanana-v-3b', source={'model': 'Kakao test fixture'})
        directory = result_directory(self.path, 'kanana-v-3b')
        write_json(directory/'automatic.json', korean)
        write_json(directory/'draft.json', korean)
        self.manager.save(self.id, self.value, 1, confirm=True)
        active_qwen = (self.path/'task_definition.json').read_bytes()
        self.manager.save(self.id, korean, 1, model_id='kanana-v-3b')
        self.assertEqual((self.path/'task_definition.json').read_bytes(), active_qwen)
        self.assertEqual(self.manager.state(self.id)['annotation']['review_status'], 'confirmed')
        with self.assertRaises(ValueError): self.manager.export(self.id, 'kanana-v-3b')
        korean = self.manager.state(self.id, 'kanana-v-3b')['annotation']
        self.manager.save(self.id, korean, 2, confirm=True, model_id='kanana-v-3b')
        self.assertEqual(self.manager.export(self.id)['model_id'], DEFAULT_MODEL)
        self.assertEqual(self.manager.export(self.id, 'kanana-v-3b')['model_id'], 'kanana-v-3b')
        self.assertEqual(read_json(self.path/'task_definition.json')['model_id'], 'kanana-v-3b')
        self.assertEqual(read_json(self.directory/'automatic.json'), self.value)
        self.assertEqual(self.manager.state(self.id, 'gemma3-4b')['job']['status'], 'idle')
        self.assertIsNone(self.manager.state(self.id, 'gemma3-4b')['annotation'])
        self.assertTrue(qwen_bytes)

    def test_invalid_model_or_cross_model_save_rejected(self):
        for model in ('../other', '', None):
            with self.subTest(model=model), self.assertRaises(ValueError):
                self.manager.state(self.id, model)
        wrong_model = dict(self.value, model_id='gemma3-4b')
        with self.assertRaisesRegex(ValueError, '출처'):
            self.manager.save(self.id, wrong_model, 1)

    def test_uninstalled_model_does_not_spawn_inference(self):
        from unittest.mock import patch
        with patch('showonce_studio.showonce_stages.model_status', return_value={'installed': False, 'setup_command': 'install fixture'}):
            with self.assertRaisesRegex(ValueError, '설치'):
                self.manager.start(self.id, 'gemma3-4b')
        self.assertIsNone(self.manager.process)

    def templated(self, model_id=DEFAULT_MODEL, indices=(2,4,6,8), protocol='shared-fixture-protocol'):
        samples = [{'frame_index': i, 'time_s': t} for i,t in enumerate(self.times)]
        output = {'object_name': 'cube', 'target_name': 'bowl', 'boundaries': {
            b['key']: {'sample_index': index, 'evidence': 'observed boundary', 'confidence': 'medium'}
            for b,index in zip(BOUNDARIES,indices)}}
        assessment = assessment_from_output(output, samples)
        value = dict(annotation_from_assessment(assessment, 1.), recording_id=self.id,
                     model_id=model_id, revision=1, review_status='unreviewed',
                     source={'model': model_id, 'protocol_id': protocol})
        directory = result_directory(self.path, model_id)
        write_json(directory/'automatic.json', value)
        write_json(directory/'draft.json', value)
        return value

    def test_fixed_template_preserves_order_rl_mapping_and_rejects_renaming(self):
        value = self.templated()
        self.assertEqual([s['code'] for s in value['stages']], ['approach','grasp','transfer','release','retreat'])
        self.assertEqual([s['rl_section'] for s in value['stages']], [0,1,2,3,4])
        for kind in ('rename','goal','count','order'):
            bad = deepcopy(value)
            if kind=='rename':bad['stages'][0]['label']='initial rest'
            if kind=='goal':bad['stages'][0]['goal']='changed definition'
            if kind=='count':bad['stages'].pop()
            if kind=='order':bad['stages'].reverse()
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                self.manager.save(self.id,bad,1)

    def test_unobserved_boundary_is_not_fabricated(self):
        samples = [{'frame_index':i,'time_s':i/10} for i in range(11)]
        raw = {'object_name':'cube','target_name':'bowl','boundaries':{
            b['key']:{'sample_index':index,'evidence':'uncertain cue','confidence':'low'}
            for b,index in zip(BOUNDARIES,(2,4,None,8))}}
        assessment=assessment_from_output(raw,samples)
        self.assertIsNone(assessment['boundaries']['release_start']['time_s'])
        with self.assertRaisesRegex(ValueError,'확인하지 못한'):
            annotation_from_assessment(assessment,1.)
        raw['boundaries']['release_start']['sample_index']=True
        with self.assertRaises(ValueError):assessment_from_output(raw,samples)
        raw['boundaries']['release_start']['sample_index']=3
        with self.assertRaisesRegex(ValueError,'순서'):
            annotation_from_assessment(assessment_from_output(raw,samples),1.)

    def test_human_reference_metrics_and_reviewed_export_times(self):
        qwen=self.templated()
        self.templated('kanana-v-3b',(3,5,7,9))
        before=self.manager.comparison(self.id)
        self.assertTrue(before['same_protocol'])
        self.assertTrue(all(v is None for v in before['mae_s'].values()))
        self.assertAlmostEqual(before['rows'][0]['spread_s'],.1)
        edited=deepcopy(qwen)
        edited['stages'][0]['end_s']=.23
        edited['stages'][1]['start_s']=.23
        self.manager.save(self.id,edited,1,confirm=True)
        after=self.manager.comparison(self.id)
        self.assertAlmostEqual(after['mae_s'][DEFAULT_MODEL],.0075)
        self.assertAlmostEqual(after['mae_s']['kanana-v-3b'],.0925)
        self.assertEqual(after['reference']['revision'],2)
        exported=self.manager.export(self.id)
        self.assertEqual(exported['boundaries']['grasp_start']['time_s'],.23)
        self.assertFalse(exported['training_ready'])
        self.assertEqual(read_json(self.directory/'automatic.json'),qwen)
        self.assertEqual(len(list((self.path/'stages/templates'/TEMPLATE_ID/'references').glob('*.json'))),1)

    def test_different_protocols_do_not_produce_accuracy_metrics(self):
        qwen=self.templated()
        self.manager.save(self.id,qwen,1,confirm=True)
        self.templated('gemma3-4b',protocol='different-samples-or-prompt')
        result=self.manager.comparison(self.id)
        self.assertFalse(result['same_protocol'])
        self.assertTrue(all(v is None for v in result['mae_s'].values()))
        self.assertTrue(all(r['spread_s'] is None for r in result['rows']))

    def test_protocol_hash_tracks_actual_prompt_and_samples(self):
        samples=[{'frame_index':i,'time_s':i/10} for i in range(11)]
        prompt=boundary_prompt(1.,'pick place',11)
        first=protocol_id(prompt,samples)
        self.assertEqual(first,protocol_id(prompt,deepcopy(samples)))
        self.assertNotEqual(first,protocol_id(prompt+'different',samples))
        samples[1]['time_s']+=.01
        self.assertNotEqual(first,protocol_id(prompt,samples))

    def test_legacy_free_form_results_are_not_used_for_fixed_template(self):
        old=self.path/'stages/automatic.json'
        write_json(old,self.value)
        (self.directory/'automatic.json').unlink()
        (self.directory/'draft.json').unlink()
        self.assertIsNone(self.manager.state(self.id)['automatic'])
        self.assertEqual(read_json(old),self.value)

    def test_display_name_updates_preserve_confirmed_files_and_boundaries(self):
        value=self.templated()
        self.manager.save(self.id,value,1,confirm=True)
        for name in ('draft.json','task_definition.json'):
            path=self.directory/name
            old=read_json(path)
            old['stages'][1]['label']='Grasp · 파지·들기'
            write_json(path,old)
        draft_before=(self.directory/'draft.json').read_bytes()
        task_before=(self.directory/'task_definition.json').read_bytes()
        state=self.manager.state(self.id)
        exported=self.manager.export(self.id)
        self.assertEqual(state['annotation']['stages'][1]['label'],'Grasp · 파지, 들기')
        self.assertEqual(exported['stages'][1]['label'],'Grasp · 파지, 들기')
        self.assertEqual(exported['revision'],2)
        self.assertEqual(exported['review_status'],'confirmed')
        self.assertEqual(state['annotation']['boundaries'],exported['boundaries'])
        self.assertEqual((self.directory/'draft.json').read_bytes(),draft_before)
        self.assertEqual((self.directory/'task_definition.json').read_bytes(),task_before)


if __name__ == '__main__':
    unittest.main()
