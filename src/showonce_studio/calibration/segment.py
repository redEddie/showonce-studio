"""SHOW-44: SAM 3 텍스트 프롬프트(objects.json의 phrase)로 물체별 마스크를 만든다.

출력: <run>/masks/<id>.png (0/255), <run>/masks/segment.json (점수, 넓이, 마스크 안 유효 depth 비율), <run>/masks/overlay.png
"""
import argparse
import pathlib
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from .common import CONFIG, latest_run, load_frame, load_json, save_json  # noqa: E402

from ..config import sam_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--checkpoint", default=str(sam_checkpoint()))
    ap.add_argument("--threshold", type=float, default=0.4, help="SAM 3 confidence threshold")
    ap.add_argument("--ids", nargs="*", default=None, help="일부 물체만")
    ap.add_argument("--objects", default=str(CONFIG / "objects_robot.json"), help="물체 목록(id, phrase)")
    ap.add_argument("--box", nargs=5, action="append", default=[], metavar=("ID", "X0", "Y0", "X1", "Y1"),
                    help="문구 대신 박스 프롬프트(px). 텍스트로 안 잡히는 부품용(예: robot_register가 투영한 Piper 베이스)")
    ap.add_argument("--prefer", nargs=5, action="append", default=[], metavar=("ID", "X0", "Y0", "X1", "Y1"),
                    help="문구 후보가 여럿이면 이 박스(px) 안에 가장 많이 들어간 후보를 고른다(예: 'robot arm'이 Go2도 잡을 때)")
    args = ap.parse_args()
    run = pathlib.Path(args.run) if args.run else latest_run()
    rgb, depth, K, meta = load_frame(run)
    objects = [] if args.box and args.objects == ap.get_default("objects") else load_json(args.objects)["objects"]
    objects = [o for o in objects if not args.ids or o["id"] in args.ids]
    boxes = {b[0]: np.array(b[1:], float) for b in args.box}
    prefer = {b[0]: np.array(b[1:], float) for b in args.prefer}
    objects += [{"id": k, "phrase": f"box {v.round().astype(int).tolist()}", "color_rgb": [255, 160, 0]} for k, v in boxes.items()]

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(checkpoint_path=args.checkpoint, load_from_HF=False)
    proc = Sam3Processor(model)
    proc.set_confidence_threshold(args.threshold)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = proc.set_image(Image.fromarray(rgb))
        outs = {}
        H, W = rgb.shape[:2]
        for o in objects:
            if o["id"] in boxes:
                x0, y0, x1, y1 = boxes[o["id"]]
                proc.reset_all_prompts(state)
                out = proc.add_geometric_prompt([(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H], True, state)
            else:
                proc.reset_all_prompts(state)
                # set_text_prompt는 state 자체를 돌려주고 다음 프롬프트가 덮어쓰므로 바로 복사한다
                out = proc.set_text_prompt(prompt=o["phrase"], state=state)
            outs[o["id"]] = {"scores": out["scores"].float().cpu(), "masks": out["masks"].cpu()}

    out_dir = run / "masks"
    out_dir.mkdir(exist_ok=True)
    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    report = {}
    for o in objects:
        out = outs[o["id"]]
        scores = out["scores"].numpy().ravel()
        if len(scores) == 0:
            report[o["id"]] = {"found": False, "phrase": o["phrase"]}
            print(f"{o['id']}: '{o['phrase']}' 검출 없음")
            continue
        i = int(np.argmax(scores))
        region = boxes.get(o["id"], prefer.get(o["id"]))
        if region is not None:  # 박스 프롬프트·선호 영역: 비슷한 물체 후보가 여럿이면 박스 안에 가장 많이 들어간 것
            x0, y0, x1, y1 = region.astype(int)
            inside = [float(out["masks"][j].squeeze().numpy()[max(y0, 0):y1, max(x0, 0):x1].sum()) for j in range(len(scores))]
            i = int(np.argmax(inside))
        mask = out["masks"][i].squeeze().numpy().astype(bool)
        valid = float((depth[mask] > 0).mean()) if mask.any() else 0.0
        cv2.imwrite(str(out_dir / f"{o['id']}.png"), mask.astype(np.uint8) * 255)
        report[o["id"]] = {"found": True, "phrase": o["phrase"], "score": round(float(scores[i]), 3),
                           "n_candidates": int(len(scores)), "area_px": int(mask.sum()), "depth_valid_ratio": round(valid, 3)}
        color = np.array(o["color_rgb"][::-1], np.uint8)
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * color).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        cv2.putText(overlay, f"{o['id']} {scores[i]:.2f}", (int(xs.min()), max(int(ys.min()) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        warn = "  ← 유효 depth 부족" if valid < 0.5 else ""
        print(f"{o['id']}: 점수 {scores[i]:.2f} (후보 {len(scores)}개), {mask.sum()} px, 유효 depth {valid:.0%}{warn}")
    cv2.imwrite(str(out_dir / "overlay.png"), overlay)
    save_json(out_dir / "segment.json", report)


if __name__ == "__main__":
    main()
