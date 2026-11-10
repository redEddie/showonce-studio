from showonce_studio.config import ROOT
#!/usr/bin/env python3
"""Offline local VLM stage inference. Never opens a camera or CAN."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from showonce_studio.showonce_stages import source_frames, validate_annotation, write_json, read_json
from showonce_studio.showonce_models import DEFAULT_MODEL, MODELS, model_spec, model_path, result_directory
from showonce_studio.showonce_task_template import (TEMPLATE_ID, boundary_prompt, output_constraint, assessment_from_output,
                                    annotation_from_assessment, protocol_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recording', type=Path, required=True)
    ap.add_argument('--model-id', choices=list(MODELS), default=DEFAULT_MODEL)
    ap.add_argument('--frames', type=int, default=24)
    args = ap.parse_args()
    if not 6 <= args.frames <= 48:
        ap.error('--frames must be between 6 and 48')
    spec = model_spec(args.model_id)
    weights = model_path(args.model_id)
    directory = result_directory(args.recording, args.model_id)
    directory.mkdir(parents=True, exist_ok=True)
    job = {'pid': os.getpid(), 'model_id': args.model_id, 'started_s': time.time()}
    def progress(status, **values):
        job.update(status=status, updated_s=time.time(), **values)
        write_json(directory/'job.json', job)
        print(json.dumps(job, ensure_ascii=False), flush=True)
    lock = None
    try:
        lock = (ROOT/'runs/.showonce-vlm.lock').open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('다른 VLM 분석이 GPU를 사용 중입니다.')
        if (directory/'automatic.json').exists():
            progress('complete', cached=True)
            return
        if (directory/'raw_response.txt').exists():
            archive = directory/'attempts'/str(time.time_ns())
            archive.mkdir(parents=True)
            for name in ('raw_response.txt', 'prompt.txt', 'sampled_frames.json', 'job.json',
                         'frame_observations.json', 'aggregation_prompt.txt', 'frame_observation_prompt.txt'):
                if (directory/name).exists():
                    shutil.copy2(directory/name, archive/name)
        progress('preparing', message='원본 영상 프레임과 시각을 준비합니다.')
        import numpy as np
        from PIL import Image
        rows, times = source_frames(args.recording)
        manifest = read_json(args.recording/'manifest.json')
        if not manifest.get('complete'):
            raise ValueError('시연 저장이 완료되지 않았습니다.')
        indices = sorted(set(np.linspace(0, len(rows)-1, min(args.frames, len(rows)), dtype=int).tolist()))
        images, content, samples = [], [], []
        for i in indices:
            image = Image.open(args.recording/rows[i]['rgb']).convert('RGB')
            image.thumbnail((512, 320))
            images.append(image)
            content.extend([{'type': 'text', 'text': f'SAMPLE index={len(samples)} time={times[i]:.3f}s:'},
                            {'type': 'image', 'image': image}])
            samples.append({'frame_index': i, 'time_s': times[i], 'rgb': rows[i]['rgb']})
        prompt = boundary_prompt(times[-1], manifest.get('task_name', ''), len(samples))
        content.append({'type': 'text', 'text': prompt})
        (directory/'prompt.txt').write_text(prompt)
        write_json(directory/'sampled_frames.json', {'sampling': 'uniform timestamped RGB frames from the source video',
                                                   'frames': samples})
        progress('loading_model', message=spec['name']+' 모델을 GPU에 로드합니다.', sampled_frames=len(indices))
        import torch
        from transformers import AutoProcessor
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU를 사용할 수 없습니다.')
        torch.cuda.reset_peak_memory_stats()
        if spec['adapter'] == 'kanana':
            from transformers import AutoModelForVision2Seq
            processor = AutoProcessor.from_pretrained(str(weights), local_files_only=True, trust_remote_code=True)
            model = AutoModelForVision2Seq.from_pretrained(str(weights), local_files_only=True,
                trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa')
        else:
            from transformers import Qwen3_5ForConditionalGeneration, Gemma3ForConditionalGeneration
            cls = Qwen3_5ForConditionalGeneration if spec['adapter'] == 'qwen35' else Gemma3ForConditionalGeneration
            processor = AutoProcessor.from_pretrained(str(weights), local_files_only=True)
            model = cls.from_pretrained(str(weights), local_files_only=True,
                dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa')
        model.eval()
        loaded = time.monotonic()
        progress('encoding', message='시각과 RGB 프레임을 인코딩합니다.')
        adapter_metadata = {'pipeline': 'direct_timestamped_images',
                            'structured_decoding': 'lm-format-enforcer==0.11.3 JSON Schema',
                            'protocol_id': protocol_id(prompt, samples),
                            'template_id': TEMPLATE_ID,
                            'sampling_max_gap_s': max(b['time_s']-a['time_s'] for a,b in zip(samples,samples[1:]))}
        if spec['adapter'] == 'kanana':
            frame_text = '\n'.join(f"SAMPLE index={i} time={sample['time_s']:.3f}s: <image>" for i,sample in enumerate(samples))
            inputs = processor.batch_encode_collate([{'image': images, 'conv': [
                {'role': 'user', 'content': frame_text+'\n'+prompt}]}],
                padding_side='left', add_generation_prompt=True, max_length=30000)
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k,v in inputs.items()}
            output_contains_prompt = False  # official image adapter generates from inputs_embeds
        else:
            messages = [{'role': 'user', 'content': content}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = processor(text=[text], images=images, return_tensors='pt').to(model.device, dtype=torch.bfloat16)
            output_contains_prompt = True
        progress('generating', message='VLM이 단계 경계와 물체 상태를 분석하고 있습니다.',
                 input_tokens=int(inputs['input_ids'].shape[1]))
        from showonce_studio.showonce_vlm_adapters import GenerationProgress, json_completion_stop
        streamer = GenerationProgress(directory, processor, progress)
        generate_kwargs = {'prefix_allowed_tokens_fn': output_constraint(processor.tokenizer, len(samples))}
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=1200, do_sample=False, use_cache=True,
                                    streamer=streamer,
                                    stopping_criteria=json_completion_stop(processor, int(inputs['input_ids'].shape[1]) if output_contains_prompt else 0),
                                    **generate_kwargs)
        generated = output[:, inputs['input_ids'].shape[1]:] if output_contains_prompt else output
        raw = processor.batch_decode(generated, skip_special_tokens=True)[0]
        (directory/'raw_response.txt').write_text(raw)
        answer = raw.strip()
        if '</think>' in answer:
            answer = answer.split('</think>')[-1].strip()
        first, last = answer.find('{'), answer.rfind('}')
        if first < 0 or last < first:
            raise ValueError('VLM이 JSON 객체를 반환하지 않았습니다. 원본 응답을 확인하세요.')
        value = json.loads(answer[first:last+1])
        assessment = assessment_from_output(value, samples)
        provenance = read_json(weights.parent/(spec['folder']+'-download.json'), {})
        result = dict(assessment, schema='showonce_stage_annotation_v2', recording_id=args.recording.name,
            model_id=args.model_id,
            revision=1, review_status='unreviewed', created_s=time.time(),
            source={'kind': 'local_vlm', 'model': spec['repo'], 'revision': provenance.get('revision', spec['revision']),
                    'license': spec['license'], 'license_url': spec['license_url'],
                    'transformers': importlib.metadata.version('transformers'), 'torch': torch.__version__,
                    'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                    'video_sha256': hashlib.sha256((args.recording/'rgb.mp4').read_bytes()).hexdigest(),
                    'sampling': 'timestamped RGB frames', 'sampled_frames': len(indices),
                    'max_new_tokens': 1200, 'do_sample': False, 'enable_thinking': False, 'use_cache': True,
                    'output_tokens': int(generated.shape[1]),
                    'inference_seconds_after_load': time.monotonic()-loaded,
                    'peak_cuda_allocated_gb': torch.cuda.max_memory_allocated()/1024**3,
                    **adapter_metadata})
        write_json(directory/'assessment.json', result)
        result = dict(result, **annotation_from_assessment(assessment, times[-1]))
        result = validate_annotation(result, times[-1], times)
        write_json(directory/'automatic.json', result)
        if not (directory/'draft.json').exists():
            write_json(directory/'draft.json', result)
            write_json(directory/'history/review-0001.json', result)
        provenance.update(inference_tested=True, tested_recording=args.recording.name,
                          tested_s=time.time(), revision=spec['revision'])
        write_json(weights.parent/(spec['folder']+'-download.json'), provenance)
        progress('complete', message='자동 분할 완료', stages=len(result['stages']),
                 elapsed_s=time.time()-job['started_s'])
    except Exception as error:
        traceback.print_exc()
        progress('error', error=f'{type(error).__name__}: {error}')
        raise SystemExit(1)
    finally:
        if lock:
            lock.close()


if __name__ == '__main__':
    main()
