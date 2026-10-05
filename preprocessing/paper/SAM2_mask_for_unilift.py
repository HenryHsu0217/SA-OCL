import argparse
import re
from pathlib import Path
import os

import numpy as np
from PIL import Image
import torch
import cv2

# --- SAM2 Imports ---
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

FRAME_RE = re.compile(r"^\d{2}\.png$")  # "00.png" ... "59.png"


def load_image_rgb(path: Path) -> np.ndarray:
    """Load image as RGB uint8 (H,W,3)."""
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def masks_to_short_id(mask_list, h: int, w: int) -> np.ndarray:
    """
    Convert SAM masks to 8-bit instance IDs (1-255).
    0 is reserved for background.
    """
    # Use uint8 to strictly limit range to 0-255
    inst = np.zeros((h, w), dtype=np.uint8)
    if not mask_list:
        return inst

    # Sort by area descending so smaller objects are painted over larger ones
    # (DEVA/SAM default behavior)
    mask_list = sorted(mask_list, key=lambda m: m.get("area", 0), reverse=True)
    
    for i, m in enumerate(mask_list, start=1):
        seg = m["segmentation"]
        # If instances > 255, we wrap around. 
        # (i % 255) ensures unique colors for adjacent objects.
        val = (i % 255) 
        if val == 0: val = 255
        inst[seg] = val

    return inst


def save_mask_short_id(path: Path, mask_u8: np.ndarray) -> None:
    """
    Saves as an 8-bit indexed PNG. 
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the PIL image in 'P' (Palette) mode
    img = Image.fromarray(mask_u8, mode="L").convert("P")
    
    # Create a colorful palette
    np.random.seed(42) # Consistent colors for debugging
    palette = np.random.randint(60, 255, size=(256, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0] # Background is black
    
    img.putpalette(palette.flatten())
    img.save(path)


def build_sam2_generator(
    checkpoint: str,
    config: str,
    device: str,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    points_per_side: int,
    min_mask_region_area: float,
):
    print(f"Loading SAM2 from {checkpoint} (config: {config})...")
    
    # SAM2 model initialization
    # apply_postprocessing=False is often safer for raw mask generation pipelines
    sam2_model = build_sam2(config, checkpoint, device=device, apply_postprocessing=False)

    # Automatic mask generator
    generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )
    return generator


def iter_scene_frames(scene_dir: Path):
    """
    Yields (frame_id, frame_path) for files named XX.png, skipping directories.
    """
    for p in sorted(scene_dir.iterdir()):
        if not p.is_file():
            continue
        if p.parent.name == ".ipynb_checkpoints":
            continue
        if p.suffix.lower() != ".png":
            continue
        if not FRAME_RE.match(p.name):
            continue
        yield int(p.stem), p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", type=str, required=True,
                        help="Root image dir, e.g. data/OCTScenes_grouped/128x128/image")
    parser.add_argument("--out_root", type=str, required=True,
                        help="Root output dir, e.g. data/OCTScenes_grouped/128x128/SAM2_mask_Uni")
    parser.add_argument("--scene_start", type=int, required=True)
    parser.add_argument("--scene_end", type=int, required=True)

    # SAM2 requires both a checkpoint and a config file
    parser.add_argument("--sam_checkpoint", type=str, default="preprocessing/paper/SAM2/weights/sam2.1_hiera_base_plus.pt",
                        help="Path to SAM2 checkpoint, e.g. sam2_hiera_large.pt")
    parser.add_argument("--sam_config", type=str, default="configs/sam2.1/sam2.1_hiera_b+.yaml",
                        help="Path to SAM2 config file, e.g. sam2_hiera_l.yaml")

    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu")

    # Tuning knobs
    parser.add_argument("--pred_iou_thresh", type=float, default=0.8)
    parser.add_argument("--stability_score_thresh", type=float, default=0.95)
    parser.add_argument("--points_per_side", type=int, default=8)
    parser.add_argument("--min_mask_region_area", type=float, default=1600)

    args = parser.parse_args()

    img_root = Path(args.img_root)
    out_root = Path(args.out_root)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; falling back to CPU.")
        args.device = "cpu"

    # Setup Generator
    generator = build_sam2_generator(
        checkpoint=args.sam_checkpoint,
        config=args.sam_config,
        device=args.device,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        points_per_side=args.points_per_side,
        min_mask_region_area=args.min_mask_region_area,
    )

    # Determine dtype for autocast (bfloat16 is preferred for SAM2)
    autocast_dtype = torch.float16
    if args.device == "cuda" and torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16

    print(f"Starting processing scenes {args.scene_start} to {args.scene_end} using {autocast_dtype}...")

    for scene in range(args.scene_start, args.scene_end + 1):
        scene_dir = img_root / f"{scene:04d}"
        if not scene_dir.is_dir():
            continue

        out_scene_dir = out_root / f"{scene:04d}"
        any_frame = False

        # Pre-calculate frames to avoid empty progress bars
        frames = list(iter_scene_frames(scene_dir))
        
        for frame_id, frame_path in frames:
            any_frame = True
            out_path = out_scene_dir / f"{frame_id:02d}.png"

            # Skip if already computed
            if out_path.exists():
                continue

            img_rgb = load_image_rgb(frame_path)
            h, w = img_rgb.shape[:2]

            # SAM2 Inference
            with torch.inference_mode(), torch.autocast(args.device, dtype=autocast_dtype):
                masks = generator.generate(img_rgb)

            inst_mask = masks_to_short_id(masks, h, w)
            save_mask_short_id(out_path, inst_mask)

        if any_frame:
            print(f"[done] scene {scene:04d}")

    print("All done.")


if __name__ == "__main__":
    main()