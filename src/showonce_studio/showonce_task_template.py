"""Shared pick/place semantics and boundary-only model comparison. No GPU/CAN."""
from copy import deepcopy
import hashlib
import json

TEMPLATE_ID = 'pick_place_v1'
STAGES = (
    {'code': 'approach', 'label': 'Approach · 접근', 'rl_section': 0,
     'goal': '손이 물체 가까이에 도달하여 파지를 시작할 준비가 됨'},
    {'code': 'grasp', 'label': 'Grasp · 파지, 들기', 'rl_section': 1,
     'goal': '물체를 확보하고 원래 지지면에서 들어 올림'},
    {'code': 'transfer', 'label': 'Transfer · 운반', 'rl_section': 2,
     'goal': '물체를 잡은 채 목표 용기 위에 도달함'},
    {'code': 'release', 'label': 'Release · 개방, 내려놓기', 'rl_section': 3,
     'goal': '목표 위치에 물체를 내려놓고 손의 파지를 해제함'},
    {'code': 'retreat', 'label': 'Retreat · 후퇴', 'rl_section': 4,
     'goal': '물체는 목표 위치에 남고 손은 물체와 용기에서 물러남'},
)
BOUNDARIES = (
    {'key': 'grasp_start', 'label': '접근 → 파지',
     'definition': 'The hand has reached close to the object and begins grasping contact/closure; not the start of reaching from far away.'},
    {'key': 'transfer_start', 'label': '파지, 들기 → 운반',
     'definition': 'A secure grasp has lifted the object clear of its original support. The object follows the hand while held.'},
    {'key': 'release_start', 'label': '운반 → 개방, 내려놓기',
     'definition': 'While still held, the object reaches above the target container and placement/lowering starts. This is before complete detachment.'},
    {'key': 'retreat_start', 'label': '개방, 내려놓기 → 후퇴',
     'definition': 'The hand has released/detached from the object at the target and can withdraw independently. The object no longer follows the hand.'},
)


def template_spec():
    return {'id': TEMPLATE_ID, 'version': 1, 'name': 'Pick & Place · SHOW-103 5단계',
            'stages': deepcopy(list(STAGES)), 'boundaries': deepcopy(list(BOUNDARIES)),
            'rl_mapping': {
                'source': 'show103-stage-reward/src/piper_sim_lab/stage_reward.py:section,main_score',
                'sections': ['approach', 'grasp_lift', 'transport', 'prepare_release', 'retreat'],
                'note': '보상 진행 구간의 의미 매핑. 4개 reset 그룹과 구별하며 정책 관측/가중치는 변경하지 않음. 영상은 접촉·높이·힘·안정 시간의 실측 판정을 대체하지 않음.'},
            'rest_policy': '처음 정지는 approach, 마지막 정지는 retreat에 포함. final position은 완료 상태.',
            'comparison_policy': '동일 영상·샘플·프롬프트·출력 형식으로 4개 경계를 비교. 사용자 확정 전에는 정답 오차를 계산하지 않음.'}


def display_annotation(value):
    """Resolve current labels without rewriting confirmed timing/history files."""
    if not value or value.get('template_id') != TEMPLATE_ID:
        return value
    result = deepcopy(value)
    names = {stage['code']: stage['label'] for stage in STAGES}
    for stage in result.get('stages', []):
        if stage.get('code') in names:
            stage['label'] = names[stage['code']]
    result['template'] = template_spec()
    return result


def boundary_prompt(duration, task_name, count):
    definitions = '\n'.join(f"- {b['key']}: {b['definition']}" for b in BOUNDARIES)
    return f'''Locate four transition boundaries of ONE human-hand pick-and-place demonstration.
Duration: {duration:.6f} seconds. There are {count} chronological RGB samples indexed 0..{count-1}.
Task hint (not evidence of success): {task_name}
The fixed task stages are approach -> grasp AND lift -> transfer -> placement/release -> retreat.
Do NOT choose the stage count, rename stages, or add initial/final-rest stages.
Initial stillness belongs to approach. Final stillness belongs to retreat; final position is an end state.
Use ONLY visible image evidence. Background robot arms are not the demonstrator.
For each boundary choose the sample_index closest to the FIRST transition into the next stage:
{definitions}
Return the sample INDEX, not a timestamp, frame number, fraction or duration.
The four indices must be strictly increasing between 1 and {count-2}, leaving a retreat interval.
Choose the nearest sample and low confidence for an uncertain visible transition.
Use null ONLY if a transition is actually absent/not visible; do not invent a successful action.
No forces, metric thresholds, 3D coordinates, or success inferred from the hint.
Return ONLY JSON with object_name, target_name and boundaries.
boundaries must have exactly grasp_start, transfer_start, release_start, retreat_start.
Every boundary needs sample_index (integer or null), evidence (brief visible cue),
and confidence (one of low, medium, high). Use brief English evidence. No markdown.
All models receive this exact task definition and these same images.
'''


def output_schema(count):
    event = {'type': 'object', 'additionalProperties': False,
             'properties': {'sample_index': {'anyOf': [{'type': 'integer', 'enum': list(range(1, count-1))}, {'type': 'null'}]},
                            'evidence': {'type': 'string', 'minLength': 1, 'maxLength': 240},
                            'confidence': {'type': 'string', 'enum': ['low', 'medium', 'high']}},
             'required': ['sample_index', 'evidence', 'confidence']}
    return {'type': 'object', 'additionalProperties': False,
            'properties': {'object_name': {'type': 'string', 'minLength': 1, 'maxLength': 80},
                           'target_name': {'type': 'string', 'minLength': 1, 'maxLength': 80},
                           'boundaries': {'type': 'object', 'additionalProperties': False,
                                          'properties': {b['key']: event for b in BOUNDARIES},
                                          'required': [b['key'] for b in BOUNDARIES]}},
            'required': ['object_name', 'target_name', 'boundaries']}


def output_constraint(tokenizer, count):
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.tokenenforcer import TokenEnforcer, TokenEnforcerTokenizerData
    # Follows lm-format-enforcer's MIT-licensed tokenizer integration, without
    # its removed Transformers 5 tokenization_utils import. The same token
    # filtering algorithm is used in both the Transformers 4 and 5 runtimes.
    token_zero = tokenizer.encode('0', add_special_tokens=False)[-1]
    special = set(tokenizer.all_special_ids)
    regular = []
    for index in range(len(tokenizer)):
        if index in special:
            continue
        after_zero = tokenizer.decode([token_zero, index])[1:]
        plain = tokenizer.decode([index])
        regular.append((index, after_zero, len(after_zero)>len(plain)))
    data = TokenEnforcerTokenizerData(regular, lambda tokens: tokenizer.decode(tokens).rstrip('�'),
                                    tokenizer.eos_token_id, False, len(tokenizer))
    enforcer = TokenEnforcer(data, JsonSchemaParser(output_schema(count)))
    def allowed_tokens(batch_id, sent):
        return enforcer.get_allowed_tokens(sent.tolist()).allowed_tokens
    return allowed_tokens


def assessment_from_output(value, samples):
    if not isinstance(value, dict) or set(value.get('boundaries', {})) != {b['key'] for b in BOUNDARIES}:
        raise ValueError('공통 템플릿의 네 경계가 모두 필요합니다.')
    result = deepcopy(value)
    for name in ('object_name', 'target_name'):
        if not isinstance(result.get(name), str) or not result[name].strip():
            raise ValueError('대상 물체·목표 용기의 이름이 필요합니다.')
    for boundary in result['boundaries'].values():
        if not isinstance(boundary, dict):
            raise ValueError('경계 정보는 객체여야 합니다.')
        index = boundary.get('sample_index')
        if index is not None and (isinstance(index, bool) or not isinstance(index, int) or not 1 <= index < len(samples)-1):
            raise ValueError('경계 샘플 번호가 영상 범위를 벗어났습니다.')
        if boundary.get('confidence') not in ('low', 'medium', 'high') or not isinstance(boundary.get('evidence'), str) or not boundary['evidence'].strip():
            raise ValueError('각 경계의 관찰 근거와 신뢰도가 필요합니다.')
        boundary['time_s'] = samples[index]['time_s'] if index is not None else None
        boundary['source_frame_index'] = samples[index]['frame_index'] if index is not None else None
    result['template_id'] = TEMPLATE_ID
    return result


def annotation_from_assessment(assessment, duration):
    events = [assessment['boundaries'][b['key']] for b in BOUNDARIES]
    if any(b['time_s'] is None for b in events):
        raise ValueError('영상에서 확인하지 못한 경계가 있습니다. 비교표의 원문 관찰을 확인하세요.')
    ends = [0.] + [b['time_s'] for b in events] + [duration]
    if any(b-a < .03 for a, b in zip(ends, ends[1:])):
        raise ValueError('모델이 공통 경계를 순서대로 구분하지 못했습니다. 원문 경계를 보존했습니다.')
    stages = []
    for i, spec in enumerate(STAGES):
        stages.append(dict(spec, id=spec['code'], start_s=ends[i], end_s=ends[i+1],
                           description=(events[i]['evidence'] if i < 4 else '해제 후 손의 후퇴와 최종 상태를 영상으로 검토하세요.'),
                           object_ids=['object', 'target']))
    return dict(assessment, template=template_spec(), objects=[
        {'id': 'object', 'name': assessment['object_name'], 'role': 'manipulated'},
        {'id': 'target', 'name': assessment['target_name'], 'role': 'target'}], stages=stages)


def validate_template(value):
    if value.get('template_id') != TEMPLATE_ID:
        raise ValueError('현재 Pick & Place 템플릿으로 다시 분석하세요.')
    stages = value.get('stages', [])
    if len(stages) != 5:
        raise ValueError('Pick & Place는 5개 고정 단계입니다. 경계만 수정하세요.')
    for stage, spec in zip(stages, STAGES):
        if any(stage.get(k) != v for k, v in spec.items()) or stage.get('id') != spec['code']:
            raise ValueError('공통 단계의 이름·순서·목표·RL 구간은 변경할 수 없습니다.')


def protocol_id(prompt, samples):
    return hashlib.sha256(json.dumps({'template': template_spec(), 'prompt': prompt, 'samples': samples,
                                     'pipeline': 'direct_timestamped_images', 'decoding': 'json_schema'},
                                    sort_keys=True, ensure_ascii=False).encode()).hexdigest()
