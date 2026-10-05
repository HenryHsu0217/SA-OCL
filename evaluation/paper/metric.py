from pathlib import Path
import os
from torchmetrics.functional import structural_similarity_index_measure as ssim
from PIL import Image
import torch
import lpips
import numpy as np
import torchvision.transforms.functional as tf
import json
from tqdm import tqdm
import torchvision.transforms as transforms
import colorsys
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from argparse import ArgumentParser
from scipy.optimize import linear_sum_assignment
from core.original.utils.image_utils import psnr
import shutil
import cv2

def iou_matrix(gt: np.ndarray, pred: np.ndarray, gt_labels, pred_labels) -> np.ndarray:
    # gt, pred are HxW or flat arrays of ints
    ious = np.zeros((len(gt_labels), len(pred_labels)), dtype=np.float64)

    for i, g in enumerate(gt_labels):
        gt_mask = (gt == g)
        gt_sum = gt_mask.sum()
        if gt_sum == 0:
            continue
        for j, p in enumerate(pred_labels):
            pred_mask = (pred == p)
            inter = np.logical_and(gt_mask, pred_mask).sum()
            union = gt_sum + pred_mask.sum() - inter
            ious[i, j] = inter / (union + 1e-12)
    return ious

def matched_miou(gt: np.ndarray, pred: np.ndarray, include_bg: bool = False) -> float:
    gt = gt.astype(np.int64)
    pred = pred.astype(np.int64)

    gt_labels = np.unique(gt)
    pred_labels = np.unique(pred)

    if not include_bg:
        gt_labels = gt_labels[gt_labels != 0]

    # No objects in GT => define mIoU as 0 (or 1, depending on convention; 0 is safer)
    if len(gt_labels) == 0:
        return 0.0

    # Compute IoU matrix
    M = iou_matrix(gt, pred, gt_labels, pred_labels)

    # Hungarian wants a cost to minimize; we want to maximize IoU
    cost = 1.0 - M
    row_ind, col_ind = linear_sum_assignment(cost)

    # Only rows correspond to gt_labels; average IoU over GT objects
    matched_ious = M[row_ind, col_ind]
    return float(matched_ious.mean())

def compute_miou_biou(gt_flat: np.ndarray, pred_flat: np.ndarray, H: int, W: int, tol: int = 1):
    gt = gt_flat.reshape(H, W)
    pred = pred_flat.reshape(H, W)


    miou = matched_miou(gt, pred, include_bg=False)
    return miou

def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)           # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5       # Alternate between 0.5 and 1.0
    l = 0.5

    
    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.uint8)
    if id==0:   #invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb

def build_rgb_to_id(max_num_obj=256):
    # LUT indexed by packed RGB = (r<<16)|(g<<8)|b
    lut = -np.ones(256**3, dtype=np.int32)

    for i in range(max_num_obj + 1):
        r, g, b = id2rgb(i, max_num_obj=max_num_obj)
        key = (int(r) << 16) | (int(g) << 8) | int(b)
        if lut[key] != -1 and lut[key] != i:
            raise ValueError(f"Color collision: id {i} and id {lut[key]} map to same RGB {r,g,b}")
        lut[key] = i

    return lut

def rgb_mask_to_ids(mask_path: Path, rgb2id_lut: np.ndarray):
    rgb = np.array(Image.open(mask_path))
    if rgb.ndim != 3:
        raise ValueError(f"Expected RGB image, got shape {rgb.shape} for {mask_path}")
    rgb = rgb[..., :3]  # drop alpha if RGBA

    packed = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2].astype(np.int32)
    ids = rgb2id_lut[packed]

    # If any pixel color is not in LUT, ids will be -1
    if (ids < 0).any():
        bad = np.unique(packed[ids < 0])
        # show first unknown color as RGB for debugging
        c = bad[0]
        r = (c >> 16) & 255
        g = (c >> 8) & 255
        b = c & 255
        raise ValueError(
            f"Found {len(bad)} unknown colors in {mask_path}. Example unknown RGB=({r},{g},{b}). "
            f"This usually happens if the mask was resized/anti-aliased/composited. "
            f"Ensure masks are saved losslessly and resized with NEAREST."
        )

    return ids.astype(np.int64)


def main(args):
    ARI_O = 0 
    ARI_A = 0
    AMI_O = 0
    AMI_A = 0
    PSNR = 0
    LPIPS = 0
    SSIM = 0
    FPS = 0
    mIoU_sum = 0.0
    loss_fn = lpips.LPIPS(net='vgg')
    count = 0
    model_path = args.model_path
    paths = model_path.split("/")
    if args.ours:
        scene = int((paths[-1]).split("_")[0])
    else:
        scene = int(paths[-1].split("_")[0])
    transform = transforms.ToTensor()
    rendered_path = os.path.join(model_path,"train")
    pred_rendered_path = os.path.join(rendered_path,"objects_pred")
    pred_rendered_path = Path(pred_rendered_path)
    GT_mask_path = os.path.join(rendered_path,"gt_ocl_mask")
    GT_image_path = os.path.join(rendered_path,"gt")
    Render_image_path = os.path.join(rendered_path,"renders")
    for rendered_pred in pred_rendered_path.iterdir():
        frame = int(rendered_pred.stem)
        GT_mask = np.array(Image.open(os.path.join(GT_mask_path, f"000{frame:02d}.png")))
        GT_image = Image.open(os.path.join(GT_image_path, f"000{frame:02d}.png")).convert("RGB")
        Render_image = Image.open(os.path.join(Render_image_path, f"000{frame:02d}.png")).convert("RGB")
        render_tensor = transform(Render_image).unsqueeze(0)
        gt_tensor = transform(GT_image).unsqueeze(0)
        psnr_v = psnr(gt_tensor, render_tensor)
        ssim_v = ssim(gt_tensor, render_tensor,data_range=1.0)

        gt = gt_tensor * 2 - 1
        render = render_tensor * 2 - 1
        lpips_v = loss_fn(render, gt)
        
        H,W = GT_mask.shape
        renderpred = np.array(Image.open(rendered_pred)).reshape(-1)
        gt_flat   = GT_mask.reshape(-1)
        pred_flat =  renderpred.reshape(-1)
        mask_fg = gt_flat != 0
        ari_O = adjusted_rand_score(gt_flat[mask_fg], pred_flat[mask_fg])
        ari_A = adjusted_rand_score(gt_flat, pred_flat)
        ami_O = adjusted_mutual_info_score(gt_flat[mask_fg], pred_flat[mask_fg])
        ami_A =  adjusted_mutual_info_score(gt_flat, pred_flat)
        miou = compute_miou_biou(gt_flat, pred_flat, H, W)
        mIoU_sum += miou
        if args.ours:
            with open(os.path.join(rendered_path, "fps.txt"), "r") as f:
                raw_fps_text = f.read().strip()
                    
                # 2. Strip out the "FPS: " part so it's just the number
            FPS = raw_fps_text.replace("FPS: ", "")

        AMI_O += ami_O
        AMI_A += ami_A
        ARI_O += ari_O
        ARI_A += ari_A
        PSNR += psnr_v
        SSIM += ssim_v
        LPIPS += lpips_v
        count += 1
    base = Path(args.output)
    out_path = base.with_name(base.name + "_train").with_suffix(".txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a+") as f:
        f.seek(0)            
        if args.ours:
            f.write(f"Scene {scene}: FG-ARI = {ARI_O/(count)}, ARI = {ARI_A/(count)}, AMI-O = {AMI_O/(count)}, AMI-A = {AMI_A/count}, mIoU = {mIoU_sum/(count)}, PSNR = {(PSNR.item())/count}, SSIM = {SSIM/count}, LPIPS = {(LPIPS.item())/count}, FPS = {FPS}\n")
        else:
            f.write(f"Scene {scene}: FG-ARI = {ARI_O/(count)}, ARI = {ARI_A/(count)}, AMI-O = {AMI_O/(count)}, AMI-A = {AMI_A/count}, mIoU = {mIoU_sum/(count)}, PSNR = {(PSNR.item())/count}, SSIM = {SSIM/count}, LPIPS = {(LPIPS.item())/count}\n")
    


    
    if args.test:
        ARI_O = 0 
        ARI_A = 0
        AMI_O = 0
        AMI_A = 0
        PSNR = 0
        LPIPS = 0
        SSIM = 0
        mIoU_sum = 0.0
        count = 0

        rendered_path = os.path.join(model_path,"test")
        transform = transforms.ToTensor()
        pred_rendered_path = os.path.join(rendered_path,"objects_pred")
        pred_rendered_path = Path(pred_rendered_path)
        GT_mask_path = os.path.join(rendered_path,"gt_ocl_mask")
        GT_image_path = os.path.join(rendered_path,"gt")
        Render_image_path = os.path.join(rendered_path,"renders")

        for rendered_pred in pred_rendered_path.iterdir():
            frame = int(rendered_pred.stem)
            GT_mask = np.array(Image.open(os.path.join(GT_mask_path, f"000{frame:02d}.png")))
            GT_image = Image.open(os.path.join(GT_image_path, f"000{frame:02d}.png")).convert("RGB")
            Render_image = Image.open(os.path.join(Render_image_path, f"000{frame:02d}.png")).convert("RGB")
            render_tensor = transform(Render_image).unsqueeze(0)
            gt_tensor = transform(GT_image).unsqueeze(0)
            psnr_v = psnr(gt_tensor, render_tensor)
            ssim_v = ssim(gt_tensor, render_tensor,data_range=1.0)

            gt = gt_tensor * 2 - 1
            render = render_tensor * 2 - 1
            lpips_v = loss_fn(render, gt)
            
            H,W = GT_mask.shape
            renderpred = np.array(Image.open(rendered_pred)).reshape(-1)
            gt_flat   = GT_mask.reshape(-1)
            pred_flat =  renderpred.reshape(-1)
            mask_fg = gt_flat != 0
            ari_O = adjusted_rand_score(gt_flat[mask_fg], pred_flat[mask_fg])
            ari_A = adjusted_rand_score(gt_flat, pred_flat)
            ami_O = adjusted_mutual_info_score(gt_flat[mask_fg], pred_flat[mask_fg])
            ami_A =  adjusted_mutual_info_score(gt_flat, pred_flat)
            miou = compute_miou_biou(gt_flat, pred_flat, H, W)
            mIoU_sum += miou
            if args.ours:
                with open(os.path.join(rendered_path, "fps.txt"), "r") as f:
                    raw_fps_text = f.read().strip()
                    
                # 2. Strip out the "FPS: " part so it's just the number
                FPS = raw_fps_text.replace("FPS: ", "")
            else:
                FPS=0

            AMI_O += ami_O
            AMI_A += ami_A
            ARI_O += ari_O
            ARI_A += ari_A
            PSNR += psnr_v
            SSIM += ssim_v
            LPIPS += lpips_v
            count += 1
        base = Path(args.output)
        out_path = base.with_name(base.name + "_test").with_suffix(".txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a+") as f:
            f.seek(0)            
            if args.ours:
                f.write(f"Scene {scene}: FG-ARI = {ARI_O/(count)}, ARI = {ARI_A/(count)}, AMI-O = {AMI_O/(count)}, AMI-A = {AMI_A/count}, mIoU = {mIoU_sum/(count)}, PSNR = {(PSNR.item())/count}, SSIM = {SSIM/count}, LPIPS = {(LPIPS.item())/count}, FPS = {FPS}\n")
            else:
                f.write(f"Scene {scene}: FG-ARI = {ARI_O/(count)}, ARI = {ARI_A/(count)}, AMI-O = {AMI_O/(count)}, AMI-A = {AMI_A/count}, mIoU = {mIoU_sum/(count)}, PSNR = {(PSNR.item())/count}, SSIM = {SSIM/count}, LPIPS = {(LPIPS.item())/count}\n")
    if not args.save:        
        shutil.rmtree(Path(model_path))

    print("Done")
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--output",type=str)
    parser.add_argument("--ours", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)
    args = parser.parse_args()   

    main(args)