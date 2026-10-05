# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
from core.paper.scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from core.paper.gaussian_renderer import render
import torchvision
from core.original.utils.general_utils import safe_state
from argparse import ArgumentParser
from core.original.arguments import ModelParams, PipelineParams, get_combined_args
from core.paper.gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import torch.nn.functional as F
import cv2
from GOLD_Code.ocl_origin import OCL
from sklearn.decomposition import PCA

def feature_to_rgb(features):
    # Input features shape: (16, H, W)
    
    # Reshape features for PCA
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    # Apply PCA and get the first 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())

    # Reshape back to (H, W, 3)
    pca_result = pca_result.reshape(H, W, 3)

    # Normalize to [0, 255]
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    rgb_array = pca_normalized.astype('uint8')

    return rgb_array

def feature_to_codebook_ids_l2(render_feat: torch.Tensor, codebook: torch.Tensor):
    """
    render_feat: (C, H, W)
    codebook:    (K, C)
    returns:     (H, W) ids
    """
    C, H, W = render_feat.shape
    x = render_feat.permute(1, 2, 0).reshape(-1, C)   # (HW, C)

    # compute squared L2 distance: ||x - e||^2 = ||x||^2 + ||e||^2 - 2 x·e
    x2 = (x * x).sum(dim=1, keepdim=True)             # (HW, 1)
    e2 = (codebook * codebook).sum(dim=1).unsqueeze(0) # (1, K)
    xe = x @ codebook.t()                              # (HW, K)

    dist2 = x2 + e2 - 2 * xe                           # (HW, K)
    ids = dist2.argmin(dim=1).reshape(H, W)
    return ids



def id2rgb(id: int):
    # reserve id=0 for black
    if id == 0:
        return np.zeros((3,), dtype=np.uint8)

    golden_ratio = 1.6180339887
    h = (id * golden_ratio) % 1.0
    s = 0.5 + (id % 2) * 0.5
    l = 0.5

    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return np.array([int(r*255), int(g*255), int(b*255)], dtype=np.uint8)

def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, codebook):
    render_path = os.path.join(model_path, name, "renders")
    gts_path = os.path.join(model_path, name,  "gt")
    colormask_path = os.path.join(model_path, name, "objects_feature16")
    gt_colormask_path = os.path.join(model_path, name, "gt_objects_color")
    pred_obj_path = os.path.join(model_path, name, "objects_pred")
    pred_obj_rgb_path = os.path.join(model_path, name, "objects_rgb_pred")
    gts_ocl = os.path.join(model_path, name,  "gt_ocl_mask")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(colormask_path, exist_ok=True)
    makedirs(gt_colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)
    makedirs(pred_obj_rgb_path, exist_ok=True)
    makedirs(gts_ocl , exist_ok=True)
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    for _ in range(10):
        _ = render(views[0], gaussians, pipeline, background)
    torch.cuda.synchronize()
    total_ms = 0.0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        starter.record()
        results = render(view, gaussians, pipeline, background)
        ender.record()
        torch.cuda.synchronize()               # ensure ender is complete
        total_ms += starter.elapsed_time(ender)
        rendering = results["render"]
        rendering_obj = results["render_object"]

        pred_obj =feature_to_codebook_ids_l2(rendering_obj, codebook).cpu().numpy().astype(np.uint8)
        pred_obj_mask = visualize_obj(pred_obj)
        

        gt_objects = view.objects
        gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8))
        gt_objects = gt_objects.to(torch.uint8).cpu().numpy()


        rgb_mask = feature_to_rgb(rendering_obj)

        Image.fromarray(pred_obj).save(os.path.join(pred_obj_path, '{0:05d}'.format(idx) + ".png"))
        Image.fromarray(rgb_mask).save(os.path.join(colormask_path, '{0:05d}'.format(idx) + ".png"))
        Image.fromarray(gt_rgb_mask).save(os.path.join(gt_colormask_path, '{0:05d}'.format(idx) + ".png"))
        Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_rgb_path, '{0:05d}'.format(idx) + ".png"))
        Image.fromarray(gt_objects).save(os.path.join(gts_ocl, '{0:05d}'.format(idx) + ".png"))
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    avg_ms = total_ms / len(views)
    fps = 1000.0 / avg_ms

    out_path = os.path.join(render_path[:-8],'concat')
    makedirs(out_path,exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*'mp4v') 
    with open(os.path.join(render_path[:-8],'fps.txt'), "a+") as f:
        f.seek(0)            
        f.write(f"FPS: {fps}\n")
    
    fps = float(5) if 'train' in out_path else float(1)
    writer = None

    for file_name in sorted(os.listdir(gts_path)):
        gt = np.array(Image.open(os.path.join(gts_path, file_name)))
        rgb = np.array(Image.open(os.path.join(render_path, file_name)))
        gt_obj = np.array(Image.open(os.path.join(gt_colormask_path, file_name)))
        render_obj = np.array(Image.open(os.path.join(colormask_path, file_name)))
        pred_obj = np.array(Image.open(os.path.join(pred_obj_rgb_path, file_name)))

        # Ensure 3-channel uint8 everywhere (defensive)
        imgs = [gt, rgb, gt_obj, pred_obj, render_obj]
        imgs = [im[..., :3].astype(np.uint8) for im in imgs]

        result = np.hstack(imgs)  # (H, 5W, 3)

        Image.fromarray(result).save(os.path.join(out_path, file_name))

        if writer is None:
            h, w = result.shape[:2]
            writer = cv2.VideoWriter(os.path.join(out_path, 'result.mp4'), fourcc, fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError("VideoWriter failed to open (codec/path issue).")

        writer.write(result[:, :, ::-1])  # RGB->BGR

    if writer is not None:
        writer.release()


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, codebook):
    with torch.no_grad():

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, codebook)

        if (not skip_test) and (len(scene.getTestCameras()) > 0):
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, codebook)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--uniflift",  action="store_true" ,default=False)
    parser.add_argument("--ours",  action="store_true" ,default=False)
    parser.add_argument("--sam2",action="store_true" ,default=False)
    parser.add_argument("--GSO", action="store_true",default=False)
    parser.add_argument("--OCT", action="store_true",default=False)
    ############################# GOCL parameters
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--img_channels', type=int, default=3)
    parser.add_argument('--data_name',default="octa")
    # Model
    parser.add_argument('--checkpoint_path', type=str)
    # DINO
    parser.add_argument('--d_dino', type=int, default=768)
    parser.add_argument('--num_dino_patches', type=int, default=784)
    # Slot Attention
    parser.add_argument('--num_iterations', type=int, default=2)
    parser.add_argument('--num_slots', type=int, default=8)
    parser.add_argument('--cnn_hidden_size', type=int, default=64)
    parser.add_argument('--slot_size', type=int, default=128)
    parser.add_argument('--mlp_hidden_size', type=int, default=512)
    # Background
    parser.add_argument('--bck_size', type=int, default=8)
    # Identity
    parser.add_argument('--int_size', type=int, default=58)
    parser.add_argument('--ext_size', type=int, default=6)
    parser.add_argument('--num_cls', type=int, default=11)
    ############################


    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    ocl_model = OCL(args)
    ocl_checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    ocl_model .load_state_dict(ocl_checkpoint )
    ocl_model .cuda()
    ocl_model .eval()
    codebook = ocl_model.ocl_encoder.protos.detach()
    bg_code = torch.zeros(122,device="cuda")
    codebook = torch.cat([bg_code[None, :], codebook], dim=0) 
    print("ocl loaded")
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, codebook )