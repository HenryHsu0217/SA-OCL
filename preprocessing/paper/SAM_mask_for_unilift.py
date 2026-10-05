import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import cv2

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


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
        # If instances > 255, we wrap around or cap. 
        # Wrapping (i % 255) ensures unique colors for adjacent objects.
        val = (i % 255) if i % 255 != 0 else 255
        inst[seg] = val

    return inst

def save_mask_short_id(path: Path, mask_u8: np.ndarray) -> None:
    """
    Saves as an 8-bit indexed PNG. 
    The data remains 1-channel grayscale (0-255), not RGB.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the PIL image in 'P' (Palette) mode
    img = Image.fromarray(mask_u8, mode="L").convert("P")
    
    # Create a colorful palette so it's not just "dark gray"
    # This is metadata only; the pixel values remain 1, 2, 3...
    palette = np.random.randint(60, 255, size=(256, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0] # Background is black
    
    img.putpalette(palette.flatten())
    img.save(path)

def build_sam_mask_generator(
    checkpoint: str,
    model_type: str,
    device: str,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    points_per_side: int,
    min_mask_region_area: int,
):
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)

    # Automatic mask generator (like DEVA's "automatic" use of SAM)
    generator = SamAutomaticMaskGenerator(
        model=sam,
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
                        help="Root output dir, e.g. data/OCTScenes_grouped/128x128/SAM_mask")
    parser.add_argument("--scene_start", type=int, required=True)
    parser.add_argument("--scene_end", type=int, required=True)

    parser.add_argument("--sam_checkpoint", type=str, required=True,
                        help="Path to SAM checkpoint, e.g. sam_vit_h_4b8939.pth")
    parser.add_argument("--model_type", type=str, default="vit_h",
                        choices=["vit_h", "vit_l", "vit_b"])

    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu")

    # Tuning knobs (similar spirit to DEVA's SAM thresholds)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.88)
    parser.add_argument("--stability_score_thresh", type=float, default=0.95)
    parser.add_argument("--points_per_side", type=int, default=32)
    parser.add_argument("--min_mask_region_area", type=int, default=0)

    args = parser.parse_args()

    img_root = Path(args.img_root)
    out_root = Path(args.out_root)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; falling back to CPU.")
        args.device = "cpu"

    generator = build_sam_mask_generator(
        checkpoint=args.sam_checkpoint,
        model_type=args.model_type,
        device=args.device,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        points_per_side=args.points_per_side,
        min_mask_region_area=args.min_mask_region_area,
    )

    for scene in range(args.scene_start, args.scene_end + 1):
        scene_dir = img_root / f"{scene:04d}"
        if not scene_dir.is_dir():
            continue

        out_scene_dir = out_root / f"{scene:04d}"
        any_frame = False

        for frame_id, frame_path in iter_scene_frames(scene_dir):
            any_frame = True
            out_path = out_scene_dir / f"{frame_id:02d}.png"

            # Skip if already computed
            if out_path.exists():
                continue

            img_rgb = load_image_rgb(frame_path)
            h, w = img_rgb.shape[:2]

            # SAM automatic segmentation
            masks = generator.generate(img_rgb)

            inst_mask = masks_to_short_id(masks, h, w)
            save_mask_short_id(out_path, inst_mask)

        if any_frame:
            print(f"[done] scene {scene:04d}")

    print("All done.")


if __name__ == "__main__":
    main()
