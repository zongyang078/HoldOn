"""Run WHAM on The Way Up videos, using the climber boxes and keypoints of the 2D line.

WHAM's own demo (``demo.py``) detects and tracks people itself (YOLOv8 + ViTPose). On The Way Up the
climber is small high on the wall, so its tracker loses them: on p3_orange it covered at most 26% of the
frames, split over three ids. This script replaces only that step. The climber's 2D keypoints come from
the 2D line's ViTPose++-H pose files (``versions.pose2d_vitpose``: same COCO-17 layout and 0-1 score
scale as WHAM's ViTPose, climber already restricted to the route region, so no belayer). Everything
after that is WHAM's own code: HMR2 image features, the WHAM network with flip averaging, SMPL.

Runs inside a WHAM checkout (Colab GPU), never in the main ``holdon`` environment. One ``.npz`` per
video, keys in external/wham/README.md. Example::

    python3 external/wham/run_wham.py --wham-root /content/WHAM --wayup data/raw/wayup \\
        --pose-dir /content/drive/MyDrive/CS5330/HoldOn/data/interim/pose2d/vitpose-plus-huge_2026-11-18 \\
        --out data/interim/wham --only p3_orange

The pure-NumPy helpers at the top (segmenting, boxes, limb vertices, alignment) import nothing from
WHAM and are unit-tested in tests/test_wham_prep.py.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
ROUTES = ("green", "orange")
LIMBS = ("left_hand", "right_hand", "left_foot", "right_foot")  # holdon.events.LIMBS

# WHAM's own thresholds (lib/models/preproc/detector.py and lib/data/datasets/dataset_custom.py).
VIS_THRESH = 0.3  # a keypoint counts as visible above this score
MINIMUM_JOINTS = 6  # a frame needs this many visible keypoints
MINIMUM_FRAMES = 30  # shorter segments are dropped
BBOX_SCALE = 1.2  # box = keypoint extent * 1.2, in WHAM's (cx, cy, size / 200) format

# SMPL joints used to pick the limb vertices (SMPL kinematic tree).
# (wrist or ankle, hand or foot, elbow or knee) per limb, in LIMBS order.
SMPL_LIMB_JOINTS = {
    "left_hand": (20, 22, 18),
    "right_hand": (21, 23, 19),
    "left_foot": (7, 10, 4),
    "right_foot": (8, 11, 5),
}


# --------------------------------------------------------------------------- pure-NumPy helpers


def frame_is_valid(scores: np.ndarray) -> np.ndarray:
    """(T,) bool: frames with at least MINIMUM_JOINTS keypoints above VIS_THRESH (as WHAM's detector)."""
    s = np.nan_to_num(scores, nan=0.0)
    return (s > VIS_THRESH).sum(axis=1) >= MINIMUM_JOINTS


def find_segments(valid: np.ndarray, max_gap: int, min_len: int = MINIMUM_FRAMES) -> list[tuple[int, int]]:
    """Inclusive ``(start, end)`` runs of valid frames; gaps of <= ``max_gap`` invalid frames are bridged.

    WHAM treats a track as a contiguous sequence, so a longer gap starts a new segment. Segments
    shorter than ``min_len`` frames are dropped (WHAM drops such tracks too).
    """
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > max_gap + 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s + 1 >= min_len]


def bbox_from_keypoints(kp: np.ndarray, scores: np.ndarray, s_factor: float = BBOX_SCALE) -> np.ndarray:
    """(T, 3) WHAM boxes ``cx, cy, scale`` from visible keypoints (``scale`` = side / 200); NaN if none.

    Same rule as ``DetectionModel.compute_bboxes_from_keypoints`` in WHAM.
    """
    vis = np.nan_to_num(scores, nan=0.0) > VIS_THRESH
    out = np.full((len(kp), 3), np.nan)
    for t in range(len(kp)):
        p = kp[t, vis[t]]
        if len(p) == 0:
            continue
        x1, y1 = p.min(axis=0)
        x2, y2 = p.max(axis=0)
        out[t] = ((x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * s_factor / 200.0)
    return out


def interpolate_nan(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN rows along axis 0 (each column separately); edges take the nearest value."""
    x = np.array(x, dtype=np.float64, copy=True)
    flat = x.reshape(len(x), -1)
    t = np.arange(len(x))
    for c in range(flat.shape[1]):
        ok = ~np.isnan(flat[:, c])
        if ok.any() and not ok.all():
            flat[:, c] = np.interp(t, t[ok], flat[ok, c])
    return flat.reshape(x.shape)


def median_filter_1d(x: np.ndarray, kernel: int) -> np.ndarray:
    """Median filter along axis 0 with edge padding (``scipy.signal.medfilt`` pads with zeros instead)."""
    if kernel <= 1 or len(x) < 2:
        return x.copy()
    h = kernel // 2
    pad = np.pad(x, [(h, h)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, kernel, axis=0)
    return np.median(win, axis=-1)


def prepare_segment(kp: np.ndarray, scores: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """WHAM inputs for one segment: ``bbox (N, 3)`` and ``keypoints (N, 17, 3)`` (x, y, score).

    Invalid frames inside the segment (bridged gaps) get interpolated positions and score 0, so WHAM
    masks their keypoints but still sees an image crop. Boxes are median-smoothed as in WHAM.
    """
    valid = frame_is_valid(scores)
    s = np.where(valid[:, None], np.nan_to_num(scores, nan=0.0), 0.0)
    xy = np.where(valid[:, None, None], kp, np.nan)
    bbox = bbox_from_keypoints(np.nan_to_num(xy, nan=0.0), np.where(valid[:, None], s, 0.0))
    bbox = interpolate_nan(bbox)
    xy = interpolate_nan(xy)
    kernel = int(int(fps / 2) / 2) * 2 + 1  # 13 at 25 fps, WHAM's choice
    bbox = median_filter_1d(bbox, kernel)
    keypoints = np.concatenate([xy, s[..., None]], axis=-1)
    return bbox.astype(np.float64), keypoints.astype(np.float64)


def limb_vertex_ids(v_template: np.ndarray, j_regressor: np.ndarray, lbs_weights: np.ndarray) -> dict[str, np.ndarray]:
    """SMPL vertex indices of each limb's contact surface, from the rest pose.

    A vertex belongs to a hand (foot) if its dominant skinning joint is that side's wrist or hand
    (ankle or foot) joint and it lies beyond the wrist (ankle) along the elbow->wrist (knee->ankle)
    direction: palm and fingers; sole, heel and toes. The forearm and shin are excluded.
    """
    joints = j_regressor @ v_template
    dominant = lbs_weights.argmax(axis=1)
    out = {}
    for limb in LIMBS:
        end, distal, proximal = SMPL_LIMB_JOINTS[limb]
        direction = joints[end] - joints[proximal]
        beyond = (v_template - joints[end]) @ direction > 0
        out[limb] = np.flatnonzero(np.isin(dominant, (end, distal)) & beyond)
    return out


def scatter_to_frames(values: np.ndarray, frame_ids: np.ndarray, num_frames: int) -> np.ndarray:
    """Place per-segment rows into a (num_frames, ...) array; frames without a value are NaN."""
    out = np.full((num_frames, *values.shape[1:]), np.nan, dtype=np.float32)
    out[frame_ids] = values
    return out


def project(points: np.ndarray, focal: float, center: tuple[float, float]) -> np.ndarray:
    """Pinhole projection of camera-frame points (..., 3) to pixels (..., 2)."""
    z = points[..., 2:3]
    return points[..., :2] / z * focal + np.asarray(center)


def reprojection_error(points3d: np.ndarray, kp2d: np.ndarray, scores: np.ndarray, focal: float,
                       center: tuple[float, float], min_score: float = 0.5) -> np.ndarray:
    """(T,) median pixel distance between projected 3D COCO joints and confident 2D keypoints; NaN if none."""
    err = np.linalg.norm(project(points3d, focal, center) - kp2d, axis=-1)
    err = np.where(np.nan_to_num(scores, nan=0.0) >= min_score, err, np.nan)
    out = np.full(len(err), np.nan, dtype=np.float32)
    has = ~np.isnan(err).all(axis=1)
    out[has] = np.nanmedian(err[has], axis=1)
    return out


# --------------------------------------------------------------------------- WHAM (Colab only)


class Wham:
    """WHAM network, feature extractor and SMPL, loaded once from a WHAM checkout."""

    def __init__(self, wham_root: Path, flip_eval: bool = True):
        os.chdir(wham_root)  # WHAM resolves configs/, dataset/ and checkpoints/ relative to its root
        sys.path.insert(0, str(wham_root))
        import torch
        from configs.config import get_cfg_defaults
        from lib.models import build_body_model, build_network
        from lib.models.preproc.extractor import FeatureExtractor

        cfg = get_cfg_defaults()
        cfg.merge_from_file("configs/yamls/demo.yaml")
        cfg.FLIP_EVAL = flip_eval
        if not torch.cuda.is_available():
            sys.exit("WHAM needs a CUDA GPU (Colab: Runtime -> Change runtime type -> GPU).")
        self.cfg = cfg
        self.torch = torch
        self.smpl = build_body_model(cfg.DEVICE, cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN)
        self.network = build_network(cfg, self.smpl)
        self.network.eval()
        self.extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)
        self.checkpoint = cfg.TRAIN.CHECKPOINT
        s = self.smpl
        self.j_regressor = s.J_regressor.cpu().numpy()  # (24, 6890) SMPL joints
        self.j_regressor_coco = s.J_regressor_wham.cpu().numpy()[:17]  # WHAM's COCO-17 regressor
        self.limb_ids = limb_vertex_ids(s.v_template.cpu().numpy(), self.j_regressor, s.lbs_weights.cpu().numpy())

    def run_video(self, video: Path, tracking: dict, width: int, height: int, fps: float, length: int) -> dict:
        """WHAM on prepared segments; returns ``{segment: {frame_ids, pose, trans, betas, verts}}``.

        The body of the subject loop follows WHAM's demo.py (flip averaging, no SMPLify, local only).
        """
        from collections import defaultdict

        from lib.data.datasets import CustomDataset
        from lib.utils.imutils import avg_preds
        from lib.utils.transforms import matrix_to_axis_angle

        torch, network = self.torch, self.network
        with torch.no_grad():
            tracking = self.extractor.run(str(video), tracking)
        slam = np.zeros((length, 7))
        slam[:, 3] = 1.0  # static camera: identity rotation every frame
        dataset = CustomDataset(self.cfg, tracking, slam, width, height, fps)
        results = defaultdict(dict)
        for subj in range(len(dataset)):
            with torch.no_grad():
                if self.cfg.FLIP_EVAL:
                    _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = dataset.load_data(subj, True)
                    flipped = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel,
                                      return_y_up=True, **kwargs)
                    _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = dataset.load_data(subj)
                    pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel,
                                   return_y_up=True, **kwargs)
                    f_pose, f_shape = flipped["pose"].squeeze(0), flipped["betas"].squeeze(0)
                    pose, shape = pred["pose"].squeeze(0), pred["betas"].squeeze(0)
                    f_pose, pose = f_pose.reshape(-1, 24, 6), pose.reshape(-1, 24, 6)
                    avg_pose, avg_shape = avg_preds(pose, shape, f_pose, f_shape)
                    network.pred_pose = avg_pose.reshape(-1, 144).view_as(network.pred_pose)
                    network.pred_shape = avg_shape.view_as(network.pred_shape)
                    avg_contact = (flipped["contact"][..., [2, 3, 0, 1]] + pred["contact"]) / 2
                    network.pred_contact = avg_contact.view_as(network.pred_contact)
                    output = network.forward_smpl(**kwargs)
                    pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)
                else:
                    _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = dataset.load_data(subj)
                    pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel,
                                   return_y_up=True, **kwargs)
            body = matrix_to_axis_angle(pred["poses_body"]).cpu().numpy().reshape(-1, 69)
            root = matrix_to_axis_angle(pred["poses_root_cam"]).cpu().numpy().reshape(-1, 3)
            results[_id]["frame_ids"] = np.asarray(frame_id)
            results[_id]["pose"] = np.concatenate([root, body], axis=-1)
            results[_id]["trans"] = (pred["trans_cam"] - network.output.offset).cpu().numpy().reshape(-1, 3)
            results[_id]["betas"] = pred["betas"].cpu().squeeze(0).numpy()
            # Camera-frame vertices in metres, exactly as demo.py stores them.
            results[_id]["verts"] = (pred["verts_cam"] + pred["trans_cam"].unsqueeze(1)).cpu().numpy()
        return results


def build_tracking(kp: np.ndarray, scores: np.ndarray, segments: list[tuple[int, int]], fps: float) -> dict:
    """WHAM ``tracking_results`` (one subject per segment) from the 2D line's keypoints."""
    from collections import defaultdict

    tracking = {}
    for i, (s, e) in enumerate(segments):
        bbox, keypoints = prepare_segment(kp[s:e + 1], scores[s:e + 1], fps)
        subj = defaultdict(list)  # the extractor appends features to lists
        subj["frame_id"] = np.arange(s, e + 1)
        subj["bbox"] = bbox
        subj["keypoints"] = keypoints
        tracking[i] = subj
    return tracking


def process_video(wham: Wham, video: Path, pose_file: Path, out_file: Path, max_gap: int,
                  max_frames: int | None) -> str:
    """Run one video end to end and write ``out_file``; returns a one-line summary."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    with np.load(pose_file, allow_pickle=False) as p:
        if str(p["keypoint_set"]) != "coco17":
            raise ValueError(f"{pose_file}: expected a coco17 (ViTPose) pose file, got {p['keypoint_set']}")
        kp, scores = p["keypoints"].astype(np.float64), p["scores"].astype(np.float64)
        pose_model = str(p["model"])
    num_frames = len(kp)
    if num_frames != length:
        raise ValueError(f"{pose_file}: {num_frames} frames, video has {length}")

    valid = frame_is_valid(scores)
    if max_frames is not None:
        valid[max_frames:] = False
    segments = find_segments(valid, max_gap)
    if not segments:
        raise RuntimeError(f"{video}: no segment of >= {MINIMUM_FRAMES} frames with a visible climber")

    t0 = time.time()
    res = wham.run_video(video, build_tracking(kp, scores, segments, fps), width, height, fps, length)

    focal = float(np.hypot(width, height))  # WHAM's default intrinsics (compute_cam_intrinsics)
    center = (width / 2.0, height / 2.0)
    seg_id = np.full(num_frames, -1, dtype=np.int16)
    parts = {k: [] for k in ("frame_ids", "pose", "trans", "betas", "joints", "coco17", "limb_verts")}
    all_ids = np.concatenate([wham.limb_ids[l] for l in LIMBS])
    for i in sorted(res):
        r = res[i]
        verts = r["verts"]
        seg_id[r["frame_ids"]] = i
        parts["frame_ids"].append(r["frame_ids"])
        for k in ("pose", "trans", "betas"):
            parts[k].append(r[k])
        parts["joints"].append(np.einsum("jv,tvc->tjc", wham.j_regressor, verts))
        parts["coco17"].append(np.einsum("jv,tvc->tjc", wham.j_regressor_coco, verts))
        parts["limb_verts"].append(verts[:, all_ids])
    fids = np.concatenate(parts["frame_ids"])
    data = {k: scatter_to_frames(np.concatenate(v), fids, num_frames) for k, v in parts.items() if k != "frame_ids"}
    reproj = reprojection_error(data["coco17"], kp, scores, focal, center)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_name(out_file.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        video_id=np.array(out_file.stem),
        format_version=np.int32(FORMAT_VERSION),
        num_frames=np.int32(num_frames),
        fps=np.float32(fps), width=np.int32(width), height=np.int32(height),
        focal_length=np.float32(focal), principal_point=np.array(center, dtype=np.float32),
        valid=seg_id >= 0, segment=seg_id,
        pose=data["pose"], trans=data["trans"], betas=data["betas"],
        joints=data["joints"], coco17=data["coco17"],
        limbs=np.array(LIMBS), limb_vertex_ids=all_ids.astype(np.int32),
        limb_vertex_counts=np.array([len(wham.limb_ids[l]) for l in LIMBS], dtype=np.int32),
        limb_verts=data["limb_verts"],
        reproj_err_px=reproj,
        keypoint_source=np.array(f"{pose_model} ({pose_file.parent.name})"),
        checkpoint=np.array(wham.checkpoint),
    )
    os.replace(tmp, out_file)
    covered = int((seg_id >= 0).sum())
    return (f"{out_file.stem:12s} {num_frames:5d} frames, {len(segments):2d} segments, "
            f"covered {covered / num_frames:6.1%}, reprojection median {np.nanmedian(reproj):5.1f} px, "
            f"{time.time() - t0:5.0f} s")


def git_commit() -> str:
    """Commit of this repository (for MANIFEST.md); ``unknown`` outside a git checkout."""
    try:
        here = Path(__file__).resolve().parent
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=here, capture_output=True, text=True).stdout
        return commit + (" (uncommitted changes)" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_manifest(out_dir: Path, args: argparse.Namespace, wham: Wham, lines: list[str]) -> None:
    """Write MANIFEST.md (docs/drive_manifest_template.md); appends the per-video summary of this run."""
    path = out_dir / "MANIFEST.md"
    if not path.exists():
        counts = ", ".join(f"{l} {len(wham.limb_ids[l])}" for l in LIMBS)
        path.write_text(f"""# MANIFEST — {out_dir.name}

| Field | Value |
| --- | --- |
| Created by | <fill in> |
| Date | {dt.date.today().isoformat()} |
| Code commit | {git_commit()} |
| Producer | external/wham/run_wham.py |
| Model / weights | WHAM (github.com/yohanshin/WHAM), `{wham.checkpoint}`, HMR2 features, SMPL neutral |
| Config | WHAM `configs/yamls/demo.yaml`, flip eval {not args.no_flip}, local only (static camera), no SMPLify, max gap {args.max_gap} frames |
| Input | processed videos `data/raw/wayup/<rec>/<route>.mp4`; keypoints and climber from `{Path(args.pose_dir).name}` |
| Environment | <fill in: Colab GPU type, torch version> |

## Files

`<video_id>.npz`: per-frame SMPL body in the camera frame (metres), NaN where WHAM has no output.
Keys and conventions: external/wham/README.md. Limb vertices (count): {counts}.
Intrinsics: WHAM default, focal = sqrt(w^2 + h^2) px, principal point at the image centre.

## Notes

<fill in>

## Runs
""")
    with path.open("a") as f:
        for line in lines:
            f.write(f"    {line}\n")


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--wham-root", type=Path, default=Path("/content/WHAM"), help="WHAM checkout with weights")
    ap.add_argument("--wayup", type=Path, default=Path("data/raw/wayup"), help="installed dataset")
    ap.add_argument("--pose-dir", type=Path, required=True, help="ViTPose pose version folder (versions.pose2d_vitpose)")
    ap.add_argument("--out", type=Path, default=Path("data/interim/wham"), help="parent of the version folder")
    ap.add_argument("--name", default="wham-vitposebox", help="version name (folder = <name>_<date>)")
    ap.add_argument("--date", default=dt.date.today().isoformat(), help="version date; keep it to resume")
    ap.add_argument("--only", nargs="*", help="video ids, e.g. p3_orange (default: all 22)")
    ap.add_argument("--max-gap", type=int, default=12, help="bridge gaps up to this many frames (default 12, ~0.5 s)")
    ap.add_argument("--max-frames", type=int, help="smoke test: use only the first N frames")
    ap.add_argument("--no-flip", action="store_true", help="disable flip averaging (about 2x faster)")
    ap.add_argument("--overwrite", action="store_true", help="redo videos that already have a file")
    args = ap.parse_args()

    # Resolve user paths before WHAM changes the working directory.
    wayup, pose_dir = args.wayup.resolve(), args.pose_dir.resolve()
    out_dir = (args.out / f"{args.name}_{args.date}").resolve()
    recs = sorted(p.name for p in wayup.iterdir() if p.is_dir() and p.name.startswith("p"))
    videos = [f"{r}_{route}" for r in recs for route in ROUTES]
    if args.only:
        unknown = set(args.only) - set(videos)
        if unknown:
            sys.exit(f"unknown video ids: {sorted(unknown)}")
        videos = [v for v in videos if v in args.only]

    wham = Wham(args.wham_root.resolve(), flip_eval=not args.no_flip)
    lines = []
    for vid in videos:
        out_file = out_dir / f"{vid}.npz"
        if out_file.exists() and not args.overwrite:
            print(f"{vid}: exists, skipped")
            continue
        rec, route = vid.rsplit("_", 1)
        line = process_video(wham, wayup / rec / f"{route}.mp4", pose_dir / f"{vid}.npz", out_file,
                             args.max_gap, args.max_frames)
        print(line, flush=True)
        lines.append(line)
    if lines and args.max_frames is None:
        write_manifest(out_dir, args, wham, lines)
    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
