# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import os
import torch
from random import randint
from core.original.utils.loss_utils import l1_loss, ssim, loss_cls_3d
from core.paper.gaussian_renderer import render, network_gui
import sys
from core.paper.scene import Scene, GaussianModel
from core.original.utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from core.original.utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from core.original.arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
import json
from GOLD_Code.utils import *
from GOLD_Code.ocl_origin import OCL

def feature_to_codebook_prob(render_feat: torch.Tensor, codebook: torch.Tensor):
    """
    render_feat: (C, H, W)
    codebook:    (K, C)
    returns:     (H, W) ids
    """

    N, C = render_feat.squeeze(1).shape
    x = render_feat.squeeze(1) # (N, C)
    x_norm = F.normalize(x, p=2, dim=1, eps=1e-8)                 # (N, C)
    cb_norm = F.normalize(codebook, p=2, dim=1, eps=1e-8)  

    sims = x_norm @ cb_norm.t()    
    probs = torch.softmax(sims, dim=1) 
    return probs

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, use_wandb, args):

    ########## Load GOCL #############
    model = OCL(args)
    ocl_checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(ocl_checkpoint )
    model.cuda()
    model.eval()
    print("ocl loaded")
    ##################################


    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
       
        image, viewspace_point_tensor, visibility_filter, radii, objects = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["render_object"]
        gt_image = viewpoint_cam.original_image.cuda()
        # Object Loss
        #####################################################
        gt_image_ocl = gt_image.unsqueeze(0)
        with torch.no_grad():
            (mse, klds, reg_bck, _, masks, pi_c ,pi_logist,  kld_ext, kld_bck, kld_pi) = model(gt_image_ocl, 0.1, 1) # B, num_slots, 1, H_enc, W_enc
        mask = torch.nn.functional.interpolate(
                masks.flatten(end_dim=1), 
                size=(args.image_size, args.image_size),
                mode='bilinear'
            )   # [B*K, 1, H, W]
        masks = mask.reshape(*masks.shape[:2], 1, args.image_size, args.image_size).squeeze(0)
        masks = torch.softmax(masks/1e-8, 0)

        codebook = model.ocl_encoder.protos.detach()
        pi_c = pi_c.squeeze(0).squeeze(0)
        slot_codes = codebook[pi_c]
        bg_code = torch.zeros(122, device="cuda")
        codes_all = torch.cat([bg_code[None, :], slot_codes], dim=0) 
        codes_all = codes_all[:, :, None, None]
        if args.masked_pixel_loss:
            diff = objects.unsqueeze(0) - codes_all
            sq_dist = (diff * diff).sum(dim=1, keepdim=True)
            loss_obj = (masks * sq_dist).mean()
        else:
            target = (masks * codes_all).sum(dim=0)[:, :, :]   
            loss_obj =  torch.nn.functional.mse_loss(objects, target)

        
        #####################################################

        # Loss
        
        Ll1 = l1_loss(image, gt_image)

        loss_obj_3d = None
        if iteration % opt.reg3d_interval == 0 and args.regloss3d:
                # regularize at certain intervals
                prob_obj3d = feature_to_codebook_prob(gaussians._objects_dc, codes_all.squeeze(-1).squeeze(-1))
                loss_obj_3d = loss_cls_3d(gaussians._xyz.squeeze().detach(), prob_obj3d, opt.reg3d_k, opt.reg3d_lambda_val, opt.reg3d_max_points, opt.reg3d_sample_size)
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + loss_obj + loss_obj_3d
        else:
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + loss_obj

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "obj": f"{loss_obj.item():.7f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), loss_obj_3d, use_wandb, loss_obj)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                #torch.save(classifier.state_dict(), os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration),'classifier.pth'))

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                #cls_optimizer.step()
                #cls_optimizer.zero_grad()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def training_report(iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_obj_3d, use_wandb, loss_obj):

    if use_wandb:
        if loss_obj_3d:
            wandb.log({"train_loss_patches/l1_loss": Ll1.item(), "train_loss_patches/total_loss": loss.item(), "train_loss_patches/loss_obj_3d": loss_obj_3d.item(), "iter_time": elapsed, "iter": iteration})
        else:
            wandb.log({"train_loss_patches/l1_loss": Ll1.item(), "Obj loss": loss_obj.item(), "train_loss_patches/total_loss": loss.item(), "iter_time": elapsed, "iter": iteration})
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if use_wandb:
                        if idx < 5:
                            wandb.log({config['name'] + "_view_{}/render".format(viewpoint.image_name): [wandb.Image(image)]})
                            if iteration == testing_iterations[0]:
                                wandb.log({config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name): [wandb.Image(gt_image)]})
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if use_wandb:
                    wandb.log({config['name'] + "/loss_viewpoint - l1_loss": l1_test, config['name'] + "/loss_viewpoint - psnr": psnr_test})
        if use_wandb:
            wandb.log({"scene/opacity_histogram": scene.gaussians.get_opacity, "total_points": scene.gaussians.get_xyz.shape[0], "iter": iteration})
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000, 60_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument("--size",type=int,default=128)
    parser.add_argument("--scene_id",type=int, default=0000)
    parser.add_argument("--split_factor", type=float, default=0.8)
    parser.add_argument("--uniflift",  action="store_true" ,default=False)
    parser.add_argument("--ours",  action="store_true" ,default=False)
    parser.add_argument("--regloss3d",action="store_true" ,default=False)
    parser.add_argument("--masked_pixel_loss", action="store_true", default=False)
    # Add an argument for the configuration file
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to the configuration file")
    parser.add_argument("--use_wandb", action='store_true', default=False, help="Use wandb to record loss value")
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

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    
    


    # Read and parse the configuration file
    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        exit(1)

    args.densify_until_iter = config.get("densify_until_iter", 15000)
    args.num_classes = config.get("num_classes", 200)
    args.reg3d_interval = config.get("reg3d_interval", 2)
    args.reg3d_k = config.get("reg3d_k", 5)
    args.reg3d_lambda_val = config.get("reg3d_lambda_val", 2)
    args.reg3d_max_points = config.get("reg3d_max_points", 300000)
    args.reg3d_sample_size = config.get("reg3d_sample_size", 1000)
    
    print("Optimizing " + args.model_path)

    if args.use_wandb:
        wandb.init(project="gaussian-splatting")
        wandb.config.args = args
        wandb.run.name = args.model_path

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.use_wandb,args)

    # All done
    print("\nTraining complete.")
