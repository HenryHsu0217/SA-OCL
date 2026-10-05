import os
import argparse
import numpy as np
import cv2
import torch
import shutil
import tempfile
from PIL import Image
from pathlib import Path

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# --- 1. Saving (DEVA Palette) ---

def save_mask_short_id(path: Path, mask_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(mask_u8, mode="L").convert("P")
    
    np.random.seed(42) 
    palette = np.random.randint(60, 255, size=(256, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0] 
    
    img.putpalette(palette.flatten())
    img.save(path)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# --- Main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--reprompt_every", type=int, default=15)
    ap.add_argument("--scene_id", type=int)
    # Tuning
    ap.add_argument("--pred_iou_thresh", type=float, default=0.9)
    ap.add_argument("--stability_score_thresh", type=float, default=1)
    ap.add_argument("--min_area", type=int, default=100)
    
    # Allow large masks (OCT layers)
    ap.add_argument("--max_mask_ratio", type=float, default=0.95)
    
    ap.add_argument("--max_objects", type=int, default=255)
    
    # LOW THRESHOLD: Default is 0.0. We use -3.0 to keep "uncertain" pixels.
    ap.add_argument("--logit_thr", type=float, default=-3.0) 
    
    ap.add_argument("--model_id", type=str, default=None)
    ap.add_argument("--min_mask_region_area", type=float, default=0)

    args = ap.parse_args()

    # --- Setup ---
    frame_names = sorted([f for f in os.listdir(os.path.join(args.img_dir,str(f"{args.scene_id:04d}"))) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    
    if not frame_names: raise RuntimeError("No images found")
    
    # PNG Temp Conversion
    video_input_path = os.path.join(args.img_dir,str(f"{args.scene_id:04d}"))
    temp_dir_obj = None
    if any(f.lower().endswith(".png") for f in frame_names):
        print("PNGs detected. Converting to temp JPGs...")
        temp_dir_obj = tempfile.TemporaryDirectory()
        video_input_path = temp_dir_obj.name
        for fname in frame_names:
            img = cv2.imread(os.path.join( os.path.join(args.img_dir,str(f"{args.scene_id:04d}")), fname))
            if img is not None:
                cv2.imwrite(os.path.join(video_input_path, os.path.splitext(fname)[0]+".jpg"), img, [int(cv2.IMWRITE_JPEG_QUALITY), 98])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Building SAM2...")
    predictor = build_sam2_video_predictor(args.cfg, args.ckpt, device=device, hydra_overrides_extra=[
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",])
    sam2_img = build_sam2(args.cfg, args.ckpt, device=device)
    amg = SAM2AutomaticMaskGenerator(sam2_img, min_mask_region_area=1600)

    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=video_input_path)

            # --- Frame 0: Auto Discovery ---
            print("Discovering objects on Frame 0...")
            img0 = cv2.imread(os.path.join( os.path.join(args.img_dir,str(f"{args.scene_id:04d}")), frame_names[0]))
            H, W = img0.shape[:2]
            total_pixels = H * W
            
            masks0 = amg.generate(img0)
            kept = []
            for m in masks0:
                seg = m["segmentation"]
                
                kept.append(seg)
            
            # Sort Largest -> Smallest
            kept.sort(key=lambda x: x.sum(), reverse=True)
            if args.max_objects > 0: kept = kept[:args.max_objects]

            if not kept: raise RuntimeError("No valid objects found on Frame 0.")
            obj_ids = list(range(1, len(kept) + 1))
            print(f"Tracking {len(obj_ids)} objects.")

            # --- Initialize Tracking ---
            for oid, seg in zip(obj_ids, kept):
                # Add mask (2D Tensor)
                mask_tensor = torch.tensor(seg.astype(np.float32)).to(device)
                predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=oid,
                    mask=mask_tensor
                )

            # --- Loop ---
            final_masks = {} 
            
            # LOCK FRAME 0: Save it immediately so it can't be overwritten
            final_masks[0] = {oid: seg.astype(np.uint8) for oid, seg in zip(obj_ids, kept)}
            
            checkpoints = list(range(args.reprompt_every, len(frame_names), args.reprompt_every))
            current_start = 0

            while current_start < len(frame_names):
                next_stop = checkpoints[0] if checkpoints else len(frame_names)
                print(f"Tracking {current_start} -> {next_stop}...")

                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state, start_frame_idx=current_start):
                    
                    # SKIP OVERWRITING FRAME 0
                    if int(out_frame_idx) == 0:
                        continue

                    frame_dict = {}
                    for i, oid in enumerate(out_obj_ids):
                        # --- REMOVED GUARD & LOWERED THRESHOLD ---
                        # We use -10.0 to keep almost everything.
                        mask = (out_mask_logits[i] > args.logit_thr).detach().cpu().numpy().squeeze().astype(np.uint8)
                        
                        # NO EXPLOSION CHECK HERE. We allow it to be 100% of screen.
                        frame_dict[int(oid)] = mask
                    
                    final_masks[int(out_frame_idx)] = frame_dict

                    # Re-prompt Check
                    if checkpoints and out_frame_idx == checkpoints[0]:
                        target_frame = checkpoints.pop(0)
                        print(f"  -> Re-prompting at Frame {target_frame}")
                        
                        prompts_added = False
                        for oid in obj_ids:
                            m = frame_dict.get(oid)
                            if m is not None and m.max() > 0:
                                mask_tensor = torch.tensor(m.astype(np.float32)).to(device)
                                predictor.add_new_mask(
                                    inference_state=state,
                                    frame_idx=target_frame,
                                    obj_id=oid,
                                    mask=mask_tensor
                                )
                                prompts_added = True
                        
                        if prompts_added:
                            current_start = target_frame
                            break 
                
                if not checkpoints and out_frame_idx >= len(frame_names) - 1: break
                if out_frame_idx >= len(frame_names) - 1: break

            # --- Save ---
            print(f"Saving indexed PNGs to {args.out_dir}...")
            ensure_dir(args.out_dir)

            for f in range(len(frame_names)):
                canvas_u8 = np.zeros((H, W), dtype=np.uint8)
                frame_seg = final_masks.get(f, {})
                
                obj_list = []
                for oid in obj_ids:
                    m = frame_seg.get(oid)
                    if m is not None and m.max() > 0:
                        obj_list.append((oid, m, m.sum()))
                
                # Largest first -> Smallest paints on top
                obj_list.sort(key=lambda x: x[2], reverse=True)
                
                for oid, mask, _ in obj_list:
                    val = (oid % 255) 
                    if val == 0: val = 255
                    canvas_u8[mask > 0] = val

                # Debug Log
                uniques = np.unique(canvas_u8)
                if f % 10 == 0: print(f"  Frame {f:05d}: Found IDs {uniques}")

                save_mask_short_id(Path(os.path.join(args.out_dir,str(f"{args.scene_id:04d}"))) / f"{f:02d}.png", canvas_u8)

    finally:
        if temp_dir_obj: temp_dir_obj.cleanup()

    print(f"Done.")

if __name__ == "__main__":
    main()