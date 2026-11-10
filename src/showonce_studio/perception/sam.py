"""SAM image subprocess lifecycle extracted from the team live perception pipeline.

Only checkpoint lookup is configured locally; inference and mask selection are retained.
"""
import collections
import threading
import time
import cv2
import numpy as np
from ..config import sam_checkpoint

cv2.setNumThreads(2)

def sam_worker(conn, phrases, threshold):
    """별도 프로세스: SAM 3. 같은 프로세스에 두면 GIL을 잡아 카메라·추적 스레드가 밀렸다(물체 나이 p95 140 → 116 ms, 2026-10-22)"""
    import os
    os.environ.setdefault('TRITON_PTXAS_BLACKWELL_PATH', '/usr/local/cuda/bin/ptxas')
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    import signal
    import traceback
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C는 부모가 처리하고 None을 보내 끝낸다
    try:
        model = build_sam3_image_model(checkpoint_path=str(sam_checkpoint()), load_from_HF=False)
        proc = Sam3Processor(model)
        proc.set_confidence_threshold(threshold)
        conn.send("ready")
        while True:
            req = conn.recv()
            if req is None:
                break
            fid, rgb = req
            res = {}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = proc.set_image(Image.fromarray(rgb))
                for oid, phrase in phrases.items():
                    proc.reset_all_prompts(state)
                    out = proc.set_text_prompt(prompt=phrase, state=state)
                    scores = out["scores"].float().cpu().numpy().ravel()
                    if len(scores):
                        i = int(np.argmax(scores))
                        res[oid] = (np.packbits(out["masks"][i].squeeze().cpu().numpy().astype(bool)), float(scores[i]))
            conn.send((fid, res))
    except Exception:  # 부모가 조용히 기다리지 않게 오류를 넘긴다(예: CUBLAS_STATUS_NOT_INITIALIZED)
        conn.send(("error", traceback.format_exc()))

class Sam(threading.Thread):
    """SAM 3 프로세스에 최신 프레임을 보내고 물체마다 최고 점수 마스크를 받는다. anchors[id] = (frame_id, mask, score)"""

    def __init__(self, cam, objects, period=1.0, threshold=0.4):
        super().__init__(daemon=True)
        import multiprocessing as mp
        self.cam, self.period = cam, period
        self.conn, child = mp.get_context("spawn").Pipe()
        self.proc = mp.get_context("spawn").Process(target=sam_worker, args=(child, {o["id"]: o["phrase"] for o in objects}, threshold), daemon=True)
        self.proc.start()
        self.anchors, self.lock, self.stop, self.times, self.ready = {}, threading.Lock(), False, [], threading.Event()
        self.error = None

    def fail(self, msg):
        self.error = msg
        print(f"\n*** SAM 3 실패 — 물체를 찾을 수 없다 ***\n{msg}", flush=True)
        self.ready.set()

    def run(self):
        try:
            first = self.conn.recv()
        except EOFError:
            return self.fail("SAM 3 프로세스가 시작 중 죽었다")
        if first != "ready":
            return self.fail(first[1] if isinstance(first, tuple) else str(first))
        self.ready.set()
        last = -1
        while not self.stop:
            f = self.cam.wait_newer(last)
            if f is None or f["id"] == last:
                continue
            last = f["id"]
            t0 = time.perf_counter()
            try:
                self.conn.send((f["id"], cv2.cvtColor(f["bgr"], cv2.COLOR_BGR2RGB)))
                fid, res = self.conn.recv()
            except (EOFError, BrokenPipeError):  # 종료 중이 아니면 프로세스가 죽은 것
                if not self.stop:
                    self.fail("SAM 3 프로세스가 죽었다")
                return
            if fid == "error":
                return self.fail(res)
            shape = f["bgr"].shape[:2]
            self.times.append(time.perf_counter() - t0)
            with self.lock:
                for oid, (bits, score) in res.items():
                    self.anchors[oid] = (fid, np.unpackbits(bits)[: shape[0] * shape[1]].reshape(shape).astype(bool), score)
            if self.period:
                time.sleep(self.period)
        self.conn.send(None)
        self.proc.join(5)
