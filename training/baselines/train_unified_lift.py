# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import os
import torch
import scipy
import numpy as np
from random import randint
from core.original.utils.loss_utils import l1_loss, ssim, loss_cls_3d
from core.original.gaussian_renderer_16 import render, network_gui
import sys
from core.original.scene_16 import Scene, GaussianModel
from core.original.utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from core.original.utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from core.original.arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
import json
import colorsys

from sklearn.decomposition import PCA
pca = PCA(n_components=3, svd_solver='full')
from PIL import Image
import torchvision



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

def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask


@torch.no_grad()
def get_confience_map(feature, gt_obj):
    
    H,W = feature.shape[1],feature.shape[2]
    feature = feature.reshape(16, -1).T
    
    
    # feature =  feature / (torch.norm(feature, dim=-1, keepdim=True) + 1e-6).detach()
    gt_obj = gt_obj.reshape(-1)
    label_set = torch.unique(gt_obj)
    wh = feature.shape[0]

    # cluster_ids = torch.unique(gt_obj)
    # choose_ids = []
    # clustersize = 500
    # for cid in cluster_ids:
    #     list_ids = torch.arange(wh)[gt_obj == cid]
    #     rand_idx = torch.randint(0, len(list_ids), [clustersize])
    #     choose_ids.append(list_ids[rand_idx])
    # random_idx = torch.cat(choose_ids)
    
    
    # batchsize = 32768
    # random_idx = torch.randint(0, wh, [batchsize])
    
    # feature =
    sam_t = gt_obj #[random_idx]
    sam_o = feature #[random_idx]
    
    # print(sam_t.shape)
    # print(sam_o.shape)
    # exit()
    
    sam_o = sam_o / (torch.norm(sam_o, dim=-1, keepdim=True) + 1e-6).detach()
    
    
    # results = {'semantic': out_obj[random_idx, :]}
    # target = {'sam': gt_obj[random_idx]}
    
    
    min_pixnum = 0
    cluster_ids, cnums_all = torch.unique(sam_t, return_counts=True)
    cluster_ids = cluster_ids[cnums_all > min_pixnum]
    cnums = cnums_all[cnums_all > min_pixnum]
    cnum = cluster_ids.shape[0] # cluster number

    u_list = torch.zeros([cnum, sam_o.shape[-1]], dtype=torch.float32, device=sam_o.device)
    phi_list = torch.zeros([cnum, 1], dtype=torch.float32, device=sam_o.device)


    for i in range(cnum):
        cluster = sam_o[sam_t == cluster_ids[i], :]
        u_list[i] = torch.mean(cluster, dim=0, keepdim=True)
        phi_list[i] = torch.norm(cluster - u_list[i], dim=1, keepdim=True).sum() / (cnums[i] * torch.log(cnums[i] + 10))

    

    # tau = 0.1; phi_list[:, 0] = tau    # option 1: constant temperature
    # phi_list = phi_list * (tau / phi_list.mean())     # option 2: (PCL) too small phi causes too large num in torch.exp().
    # phi_list = (phi_list - phi_list.min()) / (phi_list.max() - phi_list.min()) * 5 + 0.1   # scale to range [0.1, 5.1]
    phi_list = torch.clip(phi_list * 0.1, min=0.1, max=1.0)
    phi_list = phi_list.detach()
    
    # ProtoNCE = torch.zeros([1], dtype=torch.float32, device=sam_o.device)
    confidence = torch.zeros([wh,1]).cuda()
    for i in range(cnum):
        cluster = sam_o[sam_t == cluster_ids[i], :]

        dist = torch.exp(torch.matmul(cluster, u_list.T) / phi_list.T)  # [N_pix, N_cluster]
        confidence[sam_t == cluster_ids[i]] = dist[:, [i]] / (dist[:, :].sum(dim=1, keepdim=True) + 1e-6)

    
    # ProtoNCE = ProtoNCE/cnum
    confidence = confidence.reshape(H,W)
    confidence[confidence> 0.2] = 1
    confidence[confidence<= 0.2] = 0.0
    # print(torch.max(confidence))
    # print(torch.min(confidence))
    return confidence

def get_contrastive_loss(feature, gt_obj):
    
    
    feature = feature.reshape(16, -1).T
    
    loss_regularization =  ((torch.norm(feature, dim=-1, keepdim=True) - 1.0) ** 2).mean()
    
    # feature =  feature / (torch.norm(feature, dim=-1, keepdim=True) + 1e-6).detach()
    gt_obj = gt_obj.reshape(-1)
    label_set = torch.unique(gt_obj)
    wh = feature.shape[0]

    # cluster_ids = torch.unique(gt_obj)
    # choose_ids = []
    # clustersize = 500
    # for cid in cluster_ids:
    #     list_ids = torch.arange(wh)[gt_obj == cid]
    #     rand_idx = torch.randint(0, len(list_ids), [clustersize])
    #     choose_ids.append(list_ids[rand_idx])
    # random_idx = torch.cat(choose_ids)
    
    
    batchsize = 32768
    random_idx = torch.randint(0, wh, [batchsize])
    
    # feature =
    sam_t = gt_obj[random_idx]
    sam_o = feature[random_idx]
    
    # print(sam_t.shape)
    # print(sam_o.shape)
    # exit()
    
    sam_o = sam_o / (torch.norm(sam_o, dim=-1, keepdim=True) + 1e-6).detach()
    
    
    # results = {'semantic': out_obj[random_idx, :]}
    # target = {'sam': gt_obj[random_idx]}
    
    
    min_pixnum = 20
    cluster_ids, cnums_all = torch.unique(sam_t, return_counts=True)
    cluster_ids = cluster_ids[cnums_all > min_pixnum]
    cnums = cnums_all[cnums_all > min_pixnum]
    cnum = cluster_ids.shape[0] # cluster number

    u_list = torch.zeros([cnum, sam_o.shape[-1]], dtype=torch.float32, device=sam_o.device)
    phi_list = torch.zeros([cnum, 1], dtype=torch.float32, device=sam_o.device)


    for i in range(cnum):
        cluster = sam_o[sam_t == cluster_ids[i], :]
        u_list[i] = torch.mean(cluster, dim=0, keepdim=True)
        phi_list[i] = torch.norm(cluster - u_list[i], dim=1, keepdim=True).sum() / (cnums[i] * torch.log(cnums[i] + 10))

    

    # tau = 0.1; phi_list[:, 0] = tau    # option 1: constant temperature
    # phi_list = phi_list * (tau / phi_list.mean())     # option 2: (PCL) too small phi causes too large num in torch.exp().
    # phi_list = (phi_list - phi_list.min()) / (phi_list.max() - phi_list.min()) * 5 + 0.1   # scale to range [0.1, 5.1]
    phi_list = torch.clip(phi_list * 10, min=0.5, max=1.0)
    phi_list = phi_list.detach()
    
    ProtoNCE = torch.zeros([1], dtype=torch.float32, device=sam_o.device)

    for i in range(cnum):
        cluster = sam_o[sam_t == cluster_ids[i], :]

        dist = torch.exp(torch.matmul(cluster, u_list.T) / phi_list.T)  # [N_pix, N_cluster]

        # if not patch_flag:

        ProtoNCE += -torch.sum(torch.log(
            dist[:, [i]] / (dist[:, :].sum(dim=1, keepdim=True) + 1e-6)
            ))
    

    ProtoNCE = ProtoNCE/cnum
    return ProtoNCE,loss_regularization
    
    

def feature_to_rgb(features):
    # Input features shape: (16, H, W)
    
    # Reshape features for PCA
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    features_reshaped =  features_reshaped / (torch.norm(features_reshaped, dim=-1, keepdim=True) + 1e-6).detach()
    # Apply PCA and get the first 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())

    # Reshape back to (H, W, 3)
    pca_result = pca_result.reshape(H, W, 3)

    # Normalize to [0, 255]
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    rgb_array = pca_normalized.astype('uint8')
    return rgb_array

@torch.no_grad()
def create_virtual_gt_with_linear_assignment(labels_gt, predicted_scores):
    labels_gt = labels_gt.reshape(-1)
    predicted_scores = predicted_scores.reshape(256,-1).T
    # print(labels_gt.shape)
    # print(predicted_scores.shape)
    labels = sorted(torch.unique(labels_gt).cpu().tolist())[:predicted_scores.shape[-1]]
    predicted_probabilities = torch.softmax(predicted_scores, dim=-1)
    cost_matrix = np.zeros([len(labels), predicted_probabilities.shape[-1]])
    for lidx, label in enumerate(labels):
        # cost_matrix[lidx, :] = -(predicted_probabilities[labels_gt == label, :].sum(dim=0) / ((labels_gt == label).sum() + 1e-4)).cpu().numpy()
        cost_matrix[lidx, :] = -(predicted_probabilities[labels_gt == label, :].sum(dim=0)).cpu().numpy()
        
    assignment = scipy.optimize.linear_sum_assignment(np.nan_to_num(cost_matrix))
    new_labels = torch.zeros_like(labels_gt)
    for aidx, lidx in enumerate(assignment[0]):
        new_labels[labels_gt == labels[lidx]] = assignment[1][aidx]
    return new_labels

def clustering_for_matching(objects, virtual_gt_labels, code_book, confidence_map):
    ## obje -1, 16 codebook: 16, K
    # cluster_ids, cnums_all = torch.unique(virtual_gt_labels, return_counts=True)
    if (confidence_map>0.5).sum()==0:
        return torch.tensor(0., device=objects.device, requires_grad=True)
    confidence_map = confidence_map.reshape(-1)
    virtual_gt_labels = virtual_gt_labels.reshape(-1)[confidence_map>0.5]
    codebook_feature = code_book[virtual_gt_labels]
    # print(codebook_feature.shape)
    objects = objects.detach().T
    # objects = objects.reshape(-1,16)
    objects = objects/(torch.norm(objects, dim=-1, keepdim=True) + 1e-6)
    objects = objects[confidence_map>0.5]
    # print(objects.shape)
    # exit()
    clustering_loss = (torch.norm(objects-codebook_feature, dim=-1, keepdim=True)).mean()
    return clustering_loss
    #
        
    
    
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, use_wandb, weight_loss):
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    num_classes = dataset.num_classes
    print("Num classes: ",num_classes)
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1, bias=False)
    # code_book = torch.randn(256, 16, requires_grad=True, device="cuda")
    code_book = torch.tensor(classifier.weight.reshape(256, 16), requires_grad=True, device="cuda")
    
    # print(code_book[:9,:9])
    cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    cls_optimizer = torch.optim.Adam([code_book], lr=5e-4)
    code_book.cuda()
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    # opt.iterations = 35000
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

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

        # Object Loss
        ## define the contrastive loss
        gt_obj = viewpoint_cam.objects.cuda().long()
        # if iteration%50==0:
        contrastive_loss,regularization= get_contrastive_loss(objects, gt_obj)
        confidence_map = get_confience_map(objects, gt_obj)
        # else:
            # contrastive_loss = 0.0
        
        # print(objects.shape)
        # exit()
        
        # objects = objects / (torch.norm(objects, dim=0, keepdim=True) + 1e-6).detach()
        
        
        # print("objects.shape", objects.shape)
        objects = objects.reshape(16, -1)
        # logits = classifier(objects)
        logits = torch.matmul(code_book, objects.detach()) 
        # pred_obj = torch.argmax(predict, dim=1)
        H,W = gt_obj.shape[0], gt_obj.shape[1]
        logits = logits.reshape(256, H,W)
        # print("logits.shape", logits.shape)
        # exit()
        # loss_obj = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
        virtual_gt_labels = create_virtual_gt_with_linear_assignment(gt_obj, logits)
        predicted_labels = logits.argmax(dim=0)
        virtual_gt_labels = virtual_gt_labels.reshape(H,W)
        
        noise_flag = (confidence_map>0.5).sum()
        if torch.any(virtual_gt_labels != predicted_labels) and noise_flag>0:  # should never reinforce correct labels
            loss_obj_cls = (((cls_criterion(logits.unsqueeze(0), virtual_gt_labels.unsqueeze(0))).squeeze())[confidence_map>0.5]).mean()
        else:
            loss_obj_cls = torch.tensor(0., device=logits.device, requires_grad=True)
        loss_obj_cls = loss_obj_cls / torch.log(torch.tensor(num_classes))  # normalize to (0,1)


        # clustering loss
        # if iteration>
        if noise_flag>0:
            clustering_loss = clustering_for_matching(objects, virtual_gt_labels, code_book,confidence_map)
        else:
            clustering_loss = torch.tensor(0., device=logits.device, requires_grad=True)
            


        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        loss_obj_3d = None
        # contrastive_loss = contrastive_loss * 1e-6 
        # # + loss_obj_cls +
        # regularization = regularization * 1e-6
        # loss_obj_cls = loss_obj_cls * 1e-4
        # 
        if False:
            # regularize at certain intervals
            logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
            prob_obj3d = torch.softmax(logits3d,dim=0).squeeze().permute(1,0)
            loss_obj_3d = loss_cls_3d(gaussians._xyz.squeeze().detach(), prob_obj3d, opt.reg3d_k, opt.reg3d_lambda_val, opt.reg3d_max_points, opt.reg3d_sample_size)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + contrastive_loss + loss_obj_3d
        else:
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + contrastive_loss * 1e-6   + loss_obj_cls * 1e-4  + clustering_loss * weight_loss +  regularization  * 1e-6

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}, Fea: {contrastive_loss.item()}, reg: {regularization}, classification: {loss_obj_cls}, clustering_loss: {clustering_loss}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(iteration, Ll1, loss, l1_loss, contrastive_loss, loss_obj_cls, regularization, clustering_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), loss_obj_3d, use_wandb)
            
            # training_report(iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), loss_obj_3d, use_wandb)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                torch.save(code_book, os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration),'classifier.pth'))
                
                
                
                # feature_rgb = feature_to_rgb
            if iteration % 500 ==0:
                objects = objects.reshape(16, H, W)
                rgb_mask = feature_to_rgb(objects)
                
                confidence_map = (confidence_map.squeeze().cpu().numpy() * 255).astype(np.uint8)
                
                
                os.makedirs(os.path.join(scene.model_path, "save_img"), exist_ok = True)
                Image.fromarray(confidence_map).save(os.path.join(scene.model_path, "save_img/confidence_Feature_iteration_{0:05d}.png".format(iteration)))       
                
                Image.fromarray(rgb_mask).save(os.path.join(scene.model_path, "save_img/PCA_Feature_iteration_{0:05d}.png".format(iteration)))
                torchvision.utils.save_image(image, os.path.join(scene.model_path, "save_img/RGB_iteration_{0:05d}.png".format(iteration)))
                
                
                pred_obj = torch.argmax(logits,dim=0)
                pred_obj_arr = pred_obj.cpu().numpy().astype(np.uint8)
                pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))
                Image.fromarray(pred_obj_mask).save(os.path.join(scene.model_path, "save_img/classification_{0:05d}.png".format(iteration)))
                

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
                old_value = code_book.detach().data.clone()
                
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                cls_optimizer.step()
                cls_optimizer.zero_grad()
                
                # print(old_value.shape)
                # print("save old value",old_value[:3,:3])
                # print("updated value",code_book[:3,:3])
                # print(code_book.shape)
                # with torch.no_grad():
                    # for param_q, param_k in zip(fastnet.parameters(), slownet.parameters()):
                        # param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)
                    # momentum = 0.1
                    # code_book.data.mul_(momentum).add_((1 - momentum) * old_value)
                    # print(code_book.shape)
                # print("move avarage value", code_book[:3,:3])
                # print(code_book[:9,:9])
                # exit()

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


def training_report(iteration, Ll1, loss, l1_loss, contrastive_loss, loss_obj_cls, regularization, clustering_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_obj_3d, use_wandb):
    if use_wandb:
        wandb.log({"train_loss_patches/l1_loss": Ll1.item(), "train_loss_patches/total_loss": loss.item(),
                   "contrastive_loss":contrastive_loss.item(), "loss_obj_cls":loss_obj_cls.item(),
                   "regularization": regularization.item(), "clustering_loss": clustering_loss.item(),
                   "iter_time": elapsed, "iter": iteration})


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
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[100, 1_000, 7_000, 20_000, 30_000, 60_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument("--size",type=int,default=128)
    parser.add_argument("--scene_id",type=int, default=0000)
    parser.add_argument("--split_factor", type=float, default=0.8)
    parser.add_argument("--uniflift",  action="store_true" ,default=False)
    parser.add_argument("--ours",  action="store_true" ,default=False)
    parser.add_argument("--sam2",  action="store_true" ,default=False)
    parser.add_argument("--GSO", action="store_true",default=False)
    parser.add_argument("--OCT", action="store_true",default=False)
    # Add an argument for the configuration file
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to the configuration file")
    parser.add_argument("--use_wandb", action='store_true', default=False, help="Use wandb to record loss value")
    parser.add_argument("--weight_loss", type=float, default=1e-0, help="Use wandb to record loss value")


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
    args.num_classes = config.get("num_classes", 256)
    args.reg3d_interval = config.get("reg3d_interval", 2)
    args.reg3d_k = config.get("reg3d_k", 5)
    args.reg3d_lambda_val = config.get("reg3d_lambda_val", 2)
    args.reg3d_max_points = config.get("reg3d_max_points", 300000)
    args.reg3d_sample_size = config.get("reg3d_sample_size", 1000)
    
    print("Optimizing " + args.model_path)
    """args.use_wandb=True

    if True:
        wandb.init(project="Unifed_Lift")
        wandb.config.args = args
        wandb.run.name = args.model_path"""

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    #network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.use_wandb, args.weight_loss)

    # All done
    print("\nTraining complete.")