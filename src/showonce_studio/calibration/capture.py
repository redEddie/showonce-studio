#!/usr/bin/env python3
"""Capture RGB-D plus passive Piper CAN feedback; never transmit robot commands.

Run in the existing perception environment (pyrealsense2, OpenCV, NumPy).
The output camera directory uses the existing RGB-D snapshot format.
Protocol units/byte order were checked against the installed Piper SDK v2
parser and feedback classes on Thor; the SDK is deliberately not imported.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import time


from piper_runtime.transports.passive_can import PassiveCAN, tx_packets


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def frame_metadata(frame) -> dict:
    result = {}
    for name, getter in (("frame_number", frame.get_frame_number),
                         ("sensor_timestamp_ms", frame.get_timestamp),
                         ("timestamp_domain", frame.get_frame_timestamp_domain)):
        try:
            value = getter()
            result[name] = str(value) if name == "timestamp_domain" else value
        except RuntimeError as error:
            result[name] = None
            result.setdefault("metadata_errors", []).append(f"{name}: {error}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing directories are refused")
    parser.add_argument("--can", default="can_piper")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0 or args.timeout_ms < 1:
        parser.error("frames and timeout must be positive; warmup must be nonnegative")
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output already exists: {args.output}")
    report = {"schema_version": 1, "output": str(args.output.resolve()),
              "can_interface": args.can, "hardware_commands_sent": False,
              "capture_started_wall_time_ns": time.time_ns(),
              "capture_started_monotonic_ns": time.monotonic_ns(),
              "tx_packets_before": tx_packets(args.can), "errors": []}
    reader = PassiveCAN(args.can)
    pipeline = None
    pipeline_started = False
    frame_host_monotonic_ns = None
    frame_host_wall_time_ns = None
    try:
        # Optional imports stay here so pure protocol tests need no camera stack.
        import cv2
        import numpy as np
        import pyrealsense2 as rs

        reader.start()
        config = rs.config()
        if args.serial:
            config.enable_device(args.serial)
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
        pipeline = rs.pipeline()
        profile = pipeline.start(config)
        pipeline_started = True
        device = profile.get_device()
        scale = device.first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        for _ in range(args.warmup):
            pipeline.wait_for_frames(args.timeout_ms)
        depth_images, frame_samples = [], []
        for _ in range(args.frames):
            frameset = pipeline.wait_for_frames(args.timeout_ms)
            frame_host_monotonic_ns = time.monotonic_ns()
            frame_host_wall_time_ns = time.time_ns()
            aligned = align.process(frameset)
            color_frame, depth_frame = aligned.get_color_frame(), aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                raise RuntimeError("Aligned RGB-D frameset is incomplete")
            color = np.asanyarray(color_frame.get_data()).copy()
            depth_images.append(np.asanyarray(depth_frame.get_data()).astype(np.float32) * scale)
            frame_samples.append({"host_monotonic_ns": frame_host_monotonic_ns,
                                  "host_wall_time_ns": frame_host_wall_time_ns,
                                  "color": frame_metadata(color_frame), "depth": frame_metadata(depth_frame)})
        intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
        stack = np.stack(depth_images)
        stack[stack == 0] = np.nan
        # All-invalid pixels are explicitly stored as zero (RealSense convention).
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            depth_m = np.nan_to_num(np.nanmedian(stack, axis=0), nan=0.0)
        depth_mm = np.clip(np.round(depth_m * 1000), 0, 65535).astype(np.uint16)
        K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        camera = args.output / "camera"
        (camera / "color").mkdir(parents=True)
        (camera / "depth").mkdir()
        if not cv2.imwrite(str(camera / "color/0.png"), color):
            raise RuntimeError("Failed to save color PNG")
        if not cv2.imwrite(str(camera / "depth/0.png"), depth_mm):
            raise RuntimeError("Failed to save depth PNG")
        np.savetxt(camera / "cam_K.txt", K)
        meta = {
            "camera": device.get_info(rs.camera_info.name),
            "serial": device.get_info(rs.camera_info.serial_number),
            "stamp": datetime.datetime.fromtimestamp(frame_host_wall_time_ns / 1e9).astimezone().isoformat(),
            "width": intr.width, "height": intr.height, "K": K.tolist(),
            "distortion_model": str(intr.model), "distortion_coeffs": list(intr.coeffs),
            "depth_scale_m": scale, "depth_frames_median": args.frames,
            "depth_valid_ratio": float((depth_mm > 0).mean()), "gravity_imu": None,
            "host_wall_time_ns": frame_host_wall_time_ns,
            "host_monotonic_ns": frame_host_monotonic_ns,
            "frame_samples": frame_samples,
            "color": frame_samples[-1]["color"],
            "depth_capture_start_host_monotonic_ns": frame_samples[0]["host_monotonic_ns"],
            "depth_capture_end_host_monotonic_ns": frame_samples[-1]["host_monotonic_ns"],
            "depth_median_note": "Depth temporal median across listed frames; color is final frame, static scene assumed.",
            "time_alignment": "Host receipt times; RealSense sensor timestamp domain is reported separately.",
        }
        save_json(camera / "meta.json", meta)
        report["camera"] = {"directory": str(camera.resolve()), "meta": "camera/meta.json",
                            "host_monotonic_ns": frame_host_monotonic_ns}
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        if pipeline_started:
            try:
                pipeline.stop()
            except Exception as error:
                report["errors"].append(f"Camera stop failed: {error}")
        reader.stop()
        feedback = reader.feedback(frame_host_monotonic_ns)
        feedback["can_interface"] = args.can
        feedback["camera_frame_host_wall_time_ns"] = frame_host_wall_time_ns
        save_json(args.output / "robot_feedback.json", feedback)
        report["telemetry_fresh_at_frame"] = feedback["telemetry_fresh_at_frame"]
        report["can_errors"] = list(reader.errors)
        report["tx_packets_after"] = tx_packets(args.can)
        before, after = report["tx_packets_before"], report["tx_packets_after"]
        report["tx_packets_delta"] = None if before is None or after is None else after - before
        report["tx_counter_note"] = "Interface-global counters include unrelated processes; this program only receives CAN frames."
        report["capture_ended_wall_time_ns"] = time.time_ns()
        report["capture_ended_monotonic_ns"] = time.monotonic_ns()
        save_json(args.output / "capture_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
