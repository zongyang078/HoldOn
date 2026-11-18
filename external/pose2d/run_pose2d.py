"""Run 2D pose estimation on The Way Up processed videos and export one ``.npz`` per video.

Runs in the separate pose environment (see README.md), not in the main holdon environment,
and deliberately does not import ``holdon``. The output format is documented in
``docs/pose2d_format.md`` and read by ``holdon.data.external.load_pose2d``.

Pipeline per frame: YOLOX person detection (rtmlib) -> pick the climber's box (on the route, tracked
over time; see select_climber) -> pose model on that single box. Both backends share the same
detector and person selection, so their outputs differ only in the pose model.

Backends:
    rtmw     RTMW whole-body (133 COCO-WholeBody keypoints incl. fingertips and toes), via rtmlib.
    vitpose  ViTPose / ViTPose++ (17 COCO keypoints), via HuggingFace transformers.

Usage:
    python3 external/pose2d/run_pose2d.py --model rtmw --device cuda
    python3 external/pose2d/run_pose2d.py --model vitpose --device cuda
    python3 external/pose2d/run_pose2d.py --model rtmw --only p1_green --max-frames 50   # smoke test
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

FORMAT_VERSION = 2
ROUTES = ("green", "orange")
RECORDING_RE = re.compile(r"p\d+[a-z]?")
ROUTE_MARGIN = 80.0  # px added around the hold boxes to form the route region
MIN_INSIDE = 0.5  # min fraction of a person box inside the route region to count as the climber
MAX_PERSONS = 8  # person boxes stored per frame (all_boxes)

# YOLOX detector shared by both backends (rtmlib "performance" preset).
YOLOX_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
)
YOLOX_INPUT = (640, 640)
RTMW_PRESETS = {
    # tag: (onnx url, input size (w, h))
    "rtmw-x-384": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
        "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
        (288, 384),
    ),
    "rtmw-x-256": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
        "rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip",
        (192, 256),
    ),
}

COCO17 = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
_FEET = ("left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel")
_FINGERS = ("thumb", "forefinger", "middle_finger", "ring_finger", "pinky_finger")


def _hand(side: str) -> tuple[str, ...]:
    return (f"{side}_hand_root",) + tuple(f"{side}_{f}{i}" for f in _FINGERS for i in range(1, 5))


# COCO-WholeBody order (MMPose names): body 17, feet 6, face 68, left hand 21, right hand 21.
WHOLEBODY133 = COCO17 + _FEET + tuple(f"face_{i}" for i in range(68)) + _hand("left") + _hand("right")
assert len(WHOLEBODY133) == 133


# --------------------------------------------------------------------------- person selection


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two ``[x1, y1, x2, y2]`` boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def route_region(annotation: Path, margin: float = ROUTE_MARGIN) -> np.ndarray:
    """Bounding box ``[x1, y1, x2, y2]`` of all hold boxes of a route, grown by ``margin`` pixels.

    Reads ``{route}_routeAnnotation.json`` (schema in docs/dataset_schema.md). The climber is always
    on the route, while belayers and bystanders stand on the floor beside it.
    """
    holds = json.loads(annotation.read_text())["holds"]
    xs = [h[c]["x"] for h in holds for c in ("topLeftCorner", "bottomRightCorner")]
    ys = [h[c]["y"] for h in holds for c in ("topLeftCorner", "bottomRightCorner")]
    return np.array([min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin], dtype=np.float32)


def fraction_inside(box: np.ndarray, region: np.ndarray) -> float:
    """Fraction of ``box`` area that lies inside ``region`` (both ``[x1, y1, x2, y2]``)."""
    ix = max(0.0, min(box[2], region[2]) - max(box[0], region[0]))
    iy = max(0.0, min(box[3], region[3]) - max(box[1], region[1]))
    area = (box[2] - box[0]) * (box[3] - box[1])
    return float(ix * iy / area) if area > 0 else 0.0


def select_climber(
    boxes: np.ndarray,
    previous: np.ndarray | None,
    region: np.ndarray | None,
    min_iou: float = 0.1,
    min_inside: float = MIN_INSIDE,
) -> int | None:
    """Pick the climber among detected person boxes, or ``None`` if no box qualifies.

    1. Only boxes with at least ``min_inside`` of their area inside the route ``region`` are
       candidates (all boxes if ``region`` is None).
    2. Follow the previous climber box: the candidate with the highest IoU if >= ``min_iou``,
       else the candidate whose centre is closest to the previous centre.
    3. Without a previous box, take the largest candidate.
    Never falling back to an off-route box is what keeps the tracker off the belayer when the
    climber is small and briefly undetected high on the wall.
    """
    if len(boxes) == 0:
        return None
    candidates = [i for i, b in enumerate(boxes) if region is None or fraction_inside(b, region) >= min_inside]
    if not candidates:
        return None
    if previous is not None:
        ious = [box_iou(boxes[i], previous) for i in candidates]
        best = int(np.argmax(ious))
        if ious[best] >= min_iou:
            return candidates[best]
        centre = (previous[:2] + previous[2:]) / 2
        dists = [np.linalg.norm((boxes[i, :2] + boxes[i, 2:]) / 2 - centre) for i in candidates]
        return candidates[int(np.argmin(dists))]
    areas = [(boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) for i in candidates]
    return candidates[int(np.argmax(areas))]


# --------------------------------------------------------------------------- backends


def require_provider(session: object, device: str) -> None:
    """Fail fast if onnxruntime silently fell back to CPU when a GPU was requested."""
    active = session.get_providers()  # type: ignore[attr-defined]
    if device.startswith("cuda") and active[0] != "CUDAExecutionProvider":
        raise SystemExit(
            f"onnxruntime is running on {active[0]}, not CUDA. Install onnxruntime-gpu (not onnxruntime) "
            "and check CUDA / cuDNN versions (see external/pose2d/README.md)."
        )

PoseFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def build_rtmw(preset: str, device: str) -> tuple[PoseFn, tuple[str, ...], str]:
    """RTMW whole-body via rtmlib. Returns (pose fn, keypoint names, model tag)."""
    from rtmlib import RTMPose

    url, size = RTMW_PRESETS[preset]
    model = RTMPose(url, model_input_size=size, backend="onnxruntime", device=device)
    require_provider(model.session, device)

    def pose(frame_bgr: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kpts, scores = model(frame_bgr, bboxes=[box.tolist()])
        return kpts[0], scores[0]

    return pose, WHOLEBODY133, preset


def build_vitpose(model_id: str, device: str) -> tuple[PoseFn, tuple[str, ...], str]:
    """ViTPose / ViTPose++ via transformers. Returns (pose fn, keypoint names, model tag)."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, VitPoseForPoseEstimation

    processor = AutoProcessor.from_pretrained(model_id)
    model = VitPoseForPoseEstimation.from_pretrained(model_id).to(device).eval()
    # ViTPose++ is a mixture-of-experts model; expert 0 is the COCO head.
    needs_index = getattr(model.config.backbone_config, "num_experts", 1) > 1

    @torch.inference_mode()
    def pose(frame_bgr: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        xywh = [[[box[0], box[1], box[2] - box[0], box[3] - box[1]]]]
        inputs = processor(image, boxes=xywh, return_tensors="pt").to(device)
        extra = {"dataset_index": torch.tensor([0], device=device)} if needs_index else {}
        outputs = model(**inputs, **extra)
        result = processor.post_process_pose_estimation(outputs, boxes=xywh)[0][0]
        return result["keypoints"].cpu().numpy(), result["scores"].cpu().numpy()

    return pose, COCO17, model_id.split("/")[-1]


# --------------------------------------------------------------------------- per-video run


def process_video(
    video: Path,
    detect: Callable[[np.ndarray], np.ndarray],
    pose: PoseFn,
    num_keypoints: int,
    max_frames: int | None,
    region: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Run detection + pose on every frame of one video and return the arrays to save."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"Cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = total if max_frames is None else min(total, max_frames)

    keypoints = np.full((n, num_keypoints, 2), np.nan, dtype=np.float32)
    scores = np.zeros((n, num_keypoints), dtype=np.float32)
    bbox = np.full((n, 4), np.nan, dtype=np.float32)
    num_persons = np.zeros(n, dtype=np.int16)
    all_boxes = np.full((n, MAX_PERSONS, 4), np.nan, dtype=np.float32)

    previous: np.ndarray | None = None
    t0 = time.time()
    index = 0
    while index < n:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = np.asarray(detect(frame), dtype=np.float32).reshape(-1, 4)
        num_persons[index] = len(boxes)
        all_boxes[index, : min(len(boxes), MAX_PERSONS)] = boxes[:MAX_PERSONS]
        pick = select_climber(boxes, previous, region)
        if pick is not None:
            previous = boxes[pick]
            kpts, conf = pose(frame, previous)
            keypoints[index] = kpts
            scores[index] = conf
            bbox[index] = previous
        index += 1
        if index % 500 == 0:
            logging.info("  %s: %d/%d frames (%.1f fps)", video.parent.name, index, n, index / (time.time() - t0))
    cap.release()
    if index < n:
        logging.warning("%s: only %d of %d frames decodable; truncating", video, index, n)
        keypoints, scores, bbox = keypoints[:index], scores[:index], bbox[:index]
        num_persons, all_boxes = num_persons[:index], all_boxes[:index]

    return {
        "keypoints": keypoints,
        "scores": scores,
        "bbox": bbox,
        "num_persons": num_persons,
        "all_boxes": all_boxes,
        "route_region": region if region is not None else np.full(4, np.nan, dtype=np.float32),
        "fps": np.float32(fps),
        "width": np.int32(width),
        "height": np.int32(height),
        "source_num_frames": np.int32(total),
    }


def list_videos(root: Path, only: set[str] | None) -> list[tuple[str, Path]]:
    """Return ``(video_id, video path)`` for every processed video under the dataset root."""
    out = []
    for rec in sorted(p.name for p in root.iterdir() if p.is_dir() and RECORDING_RE.fullmatch(p.name)):
        for route in ROUTES:
            vid = f"{rec}_{route}"
            path = root / rec / f"{route}.mp4"
            if path.exists() and (only is None or vid in only):
                out.append((vid, path))
    return out


def write_manifest(out_dir: Path, args: argparse.Namespace, tag: str) -> None:
    """Write MANIFEST.md (docs/drive_manifest_template.md) once per version folder."""
    path = out_dir / "MANIFEST.md"
    if path.exists():
        return
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        if subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip():
            commit += " (uncommitted changes)"
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    weights = RTMW_PRESETS[args.rtmw_preset][0] if args.model == "rtmw" else args.vitpose_id
    import onnxruntime

    versions = f"onnxruntime {onnxruntime.__version__}, opencv {cv2.__version__}"
    if args.model == "vitpose":
        import torch
        import transformers

        versions += f", torch {torch.__version__}, transformers {transformers.__version__}"
    path.write_text(f"""# MANIFEST — {out_dir.name}

| Field | Value |
| --- | --- |
| Created by | <fill in> |
| Date | {args.date} |
| Code commit | {commit} |
| Producer | external/pose2d/run_pose2d.py |
| Model / weights | {tag}: {weights}; detector {YOLOX_URL} |
| Config | --model {args.model} --device {args.device} --max-frames {args.max_frames} --only {args.only} |
| Input | {args.data} (wayup.zip) |
| Environment | {platform.platform()}, Python {platform.python_version()}, {versions} |

## Files

`<video_id>.npz`: per-frame 2D keypoints, see docs/pose2d_format.md (format_version {FORMAT_VERSION}).

## Notes

<fill in: failed videos, anything a teammate must know>
""")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/raw/wayup"), help="dataset root")
    parser.add_argument("--out", type=Path, default=Path("data/interim/pose2d"))
    parser.add_argument("--model", choices=("rtmw", "vitpose"), required=True)
    parser.add_argument("--rtmw-preset", choices=tuple(RTMW_PRESETS), default="rtmw-x-384")
    parser.add_argument("--vitpose-id", default="usyd-community/vitpose-plus-huge",
                        help="HuggingFace model id, e.g. usyd-community/vitpose-base-simple")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--only", nargs="*", help="video ids to process, e.g. p1_green p2a_orange")
    parser.add_argument("--max-frames", type=int, help="process only the first N frames (smoke test)")
    parser.add_argument("--overwrite", action="store_true", help="redo videos that already have output")
    parser.add_argument("--no-route-filter", action="store_true",
                        help="do not restrict the climber to the route region from *_routeAnnotation.json")
    parser.add_argument("--date", default=dt.date.today().isoformat(),
                        help="version date in the output folder name <model>_<date>; reuse it to resume a run")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import onnxruntime

    if args.device.startswith("cuda") and hasattr(onnxruntime, "preload_dlls"):
        onnxruntime.preload_dlls()  # load CUDA / cuDNN from the nvidia-* pip packages if present
    from rtmlib import YOLOX

    detector = YOLOX(YOLOX_URL, model_input_size=YOLOX_INPUT, backend="onnxruntime", device=args.device)
    require_provider(detector.session, args.device)
    if args.model == "rtmw":
        pose, names, tag = build_rtmw(args.rtmw_preset, args.device)
        keypoint_set = "coco_wholebody133"
    else:
        pose, names, tag = build_vitpose(args.vitpose_id, args.device)
        keypoint_set = "coco17"

    out_dir = args.out / f"{tag}_{args.date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(out_dir, args, tag)
    videos = list_videos(args.data, set(args.only) if args.only else None)
    if not videos:
        raise SystemExit(f"No videos found under {args.data}")

    for vid, path in videos:
        target = out_dir / f"{vid}.npz"
        if target.exists() and not args.overwrite:
            with np.load(target) as existing:
                version = int(existing["format_version"])
            if version == FORMAT_VERSION:
                logging.info("skip %s (exists)", vid)
                continue
            logging.info("%s: existing file has format_version %d, redoing", vid, version)
        logging.info("%s -> %s", vid, target)
        annotation = path.with_name(f"{path.stem}_routeAnnotation.json")
        region = route_region(annotation) if annotation.exists() and not args.no_route_filter else None
        if region is None:
            logging.warning("  %s: no route filter (climber = largest / tracked box)", vid)
        arrays = process_video(path, detector, pose, len(names), args.max_frames, region)
        missing = int(np.isnan(arrays["bbox"][:, 0]).sum())
        logging.info("  %s: %d frames, %d without a person", vid, len(arrays["bbox"]), missing)
        tmp = target.with_name(target.stem + ".tmp.npz")
        np.savez_compressed(
            tmp,
            **arrays,
            format_version=np.int32(FORMAT_VERSION),
            video_id=np.str_(vid),
            model=np.str_(tag),
            detector=np.str_("yolox_m_humanart"),
            keypoint_set=np.str_(keypoint_set),
            keypoint_names=np.array(names),
        )
        tmp.replace(target)  # atomic: a crashed run never leaves a half-written file


if __name__ == "__main__":
    main()
