"""Model-specific input adapters; all observations remain attributable to the VLM."""
import json


def json_completion_stop(processor, prompt_length):
    from transformers import StoppingCriteria, StoppingCriteriaList
    class CompleteObject(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            text = processor.batch_decode(input_ids[:, prompt_length:], skip_special_tokens=True)[0].strip()
            if not text.startswith('{') or not text.endswith('}'):
                return False
            try:
                return isinstance(json.loads(text), dict)
            except ValueError:
                return False
    return StoppingCriteriaList([CompleteObject()])


class GenerationProgress:
    """HF generation streamer reporting liveness without logging private frames."""
    def __init__(self, directory, processor, progress):
        self.directory, self.processor, self.progress = directory, processor, progress
        self.tokens = []
        self.skip_prompt = True

    def put(self, value):
        if self.skip_prompt:
            self.skip_prompt = False
            return
        self.tokens.extend(value.reshape(-1).tolist())
        if len(self.tokens) % 64 == 0:
            partial = self.processor.batch_decode([self.tokens], skip_special_tokens=True)[0]
            (self.directory/'partial_response.txt').write_text(partial)
            self.progress('generating', generated_tokens=len(self.tokens), message='단계 JSON을 생성하고 있습니다.')

    def end(self):
        self.progress('generating', generated_tokens=len(self.tokens))

