import math
import traceback, os
import argparse
from torchvision import transforms
import torch
import torchvision.utils as vutils
from tqdm import tqdm
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import InterpolationMode
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import random
import yaml
import sys
from GOLD_Code.ocl_origin import OCL
from GOLD_Code.data_img_h5 import GlobVideoDataset, OCTScenesDataset,ReplicaDataset
from GOLD_Code.utils import cosine_anneal, linear_warmup
from torch.utils.data import Dataset
from torchvision.utils import save_image, make_grid

# from torchsummary import summary 
from GOLD_Code.metrics import *
def imagenet_unnormalize(x):
    # x: (B,3,H,W) normalized
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
    return x * std + mean
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--img_channels', type=int, default=3)
parser.add_argument('--num_vis',type=int, default=4)
# Model
parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
parser.add_argument('--data_path', default='../data/gso.h5')
parser.add_argument('--data_name',)
parser.add_argument('--model_name', )
# Training
parser.add_argument('--lr_enc', type=float, default=4e-5)
parser.add_argument('--lr_dec', type=float, default=4e-4)
parser.add_argument('--lr_warmup_steps', type=int, default=10000)
parser.add_argument('--lr_half_life', type=int, default=100000)
parser.add_argument('--clip', type=float, default=0.05)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--steps', type=int, default=500000)
parser.add_argument('--beta',type=int, default=1)
# DINO
parser.add_argument('--d_dino', type=int, default=768)
parser.add_argument('--num_dino_patches', type=int, default=784)
# Slot Attention
parser.add_argument('--num_iterations', type=int, default=3)
parser.add_argument('--num_slots', type=int, default=8)
parser.add_argument('--cnn_hidden_size', type=int, default=64)
parser.add_argument('--slot_size', type=int, default=64)
parser.add_argument('--mlp_hidden_size', type=int, default=512)
# Background
parser.add_argument('--bck_size', type=int, default=8)
parser.add_argument('--reg_start', type=float, default=0.1)
parser.add_argument('--reg_final', type=float, default=0.1)
parser.add_argument('--reg_steps', type=int, default=300000)

# Identity
parser.add_argument('--int_size', type=int, default=58)
parser.add_argument('--ext_size', type=int, default=6)
parser.add_argument('--num_cls', type=int, default=10)
parser.add_argument('--coef_pi', type=float, default=0.0002)
# parser.add_argument('--gumbel', action='store_true')

parser.add_argument('--tau_start', type=float, default=1.0)
parser.add_argument('--tau_final', type=float, default=0.1)
parser.add_argument('--tau_start_steps', type=int, default=0)
parser.add_argument('--tau_final_steps', type=int, default=30000)


parser.add_argument('--kld_start', type=float, default=1.0)
parser.add_argument('--kld_final', type=float, default=1.0)
parser.add_argument('--kld_start_steps', type=int, default=30000)
parser.add_argument('--kld_final_steps', type=int, default=80000)

parser.add_argument('--local_rank', default=-1)

args = parser.parse_args()


dist.init_process_group(backend='nccl')
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

if args.seed is None:
    args.seed = random.randint(0, 0xffffffff)
torch.manual_seed(args.seed)

log_path = os.path.join('./logs', args.data_name, args.model_name)
if os.path.exists(log_path):
    print('{} had been created'.format(log_path))
else:
    os.makedirs(log_path)


if local_rank == 0:
    arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
    arg_str = '__'.join(arg_str_list)
    writer = SummaryWriter(log_path)
    writer.add_text('hparams', arg_str)

# Loading Dataset
if args.data_name == 'octa':
    train_dataset = OCTScenesDataset(
        images_root=args.data_path,
        scene_min=0,
        scene_max=1899,
        mask_root=None,  # train has no masks (as you described)
        image_transform=transforms.ToTensor(),
    )
    val_dataset = OCTScenesDataset(
        images_root=args.data_path,
        scene_min=1900,
        scene_max=1999 ,
        mask_root="data/GSO_new/128x128/GT_mask",
        image_transform=transforms.ToTensor(),
        mask_transform= transforms.Compose([                
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.long())]), 
    )
elif args.data_name == 'replica':
    train_dataset = ReplicaDataset(
        root=args.data_path,
    )
    val_dataset = ReplicaDataset(
        root=args.data_path,
        mask_=True
    )

if local_rank == 0:
    print(f'Loading {len(train_dataset)} videos for training...')
    print(f'Loading {len(val_dataset)} videos for validation...')
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': None,
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': True,
}
train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)
train_epoch_size = len(train_loader)
val_epoch_size = len(val_loader)
log_interval = train_epoch_size // (train_epoch_size*0.1)

# Load Model
model = OCL(args)
if os.path.isfile(args.checkpoint_path):
    # load_path = os.path.join(log_path, args.checkpoint_path)
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    best_epoch = checkpoint['best_epoch']
    model.load_state_dict(checkpoint['model'])
    if local_rank == 0:
        print(f'Load trained model from {args.checkpoint_path}')
else:
    checkpoint = None
    start_epoch = 0
    best_val_loss = -math.inf
    best_epoch = 0
    if local_rank == 0:
        print('Starting training ...')
model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
optimizer = Adam([
    {'params': (x[1] for x in model.named_parameters() if 'ocl_encoder' in x[0]), 'lr': 0.0},
    {'params': (x[1] for x in model.named_parameters() if 'ocl_decoder' in x[0]), 'lr': 0.0},
])

if checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])

global_steps = 0
disable_tqdm = (not sys.stdout.isatty()) or (local_rank != 0)
global_bar = tqdm(
        total=args.epochs*len(train_loader),
        disable=not local_rank == 0,
        desc="Training (all epochs)",
        unit="batch",
        position=0
    )
# Training
for epoch in range(start_epoch, args.epochs):
    model.train()

    train_sampler.set_epoch(epoch)
    for batch, img in enumerate(train_loader):
        
        global_step = epoch * train_epoch_size + batch

        coef_reg = cosine_anneal( global_step, args.reg_start, args.reg_final, 0., args.reg_steps)
        jump = linear_warmup(global_step, 0.0, 1.0,args.kld_start_steps, args.kld_start_steps + 1)
        coef_kld = linear_warmup( global_step, args.kld_start, args.kld_final, args.kld_start_steps, args.kld_final_steps)
        coef_beta = linear_warmup( global_step, 1, 1, 10000, 60000)
        tau = cosine_anneal( global_step, args.tau_start, args.tau_final, args.tau_start_steps, args.tau_final_steps)

        lr_warmup_factor_enc = linear_warmup( global_step, 0., 1.0, 0., args.lr_warmup_steps)
        lr_warmup_factor_dec = linear_warmup( global_step, 0., 1.0, 0, args.lr_warmup_steps)
        lr_decay_factor = math.exp(global_step / args.lr_half_life * math.log(0.5))
        optimizer.param_groups[0]['lr'] = lr_decay_factor * lr_warmup_factor_enc * args.lr_enc
        optimizer.param_groups[1]['lr'] = lr_decay_factor * lr_warmup_factor_dec * args.lr_dec

        img = img.cuda()

        optimizer.zero_grad()
        (mse, klds, reg_bck, _, mask,_,pi_logist,  kld_ext, kld_bck, kld_pi) = model(img, tau,coef_beta)
        loss = mse + coef_kld * klds + coef_reg * reg_bck

        p = torch.softmax(pi_logist.detach(), dim=-1) 
            # Mean usage of each class across batch+slots
        p_mean = p.mean(dim=(0, 1))                     # [C]
        # How confident assignments are
        pi_maxprob = p.max(dim=-1).values.mean().item()
        # Entropy (0 = fully collapsed)
        pi_entropy = (-(p.clamp_min(1e-9) * p.clamp_min(1e-9).log())
                    .sum(dim=-1)).mean().item()
        # How many classes are actually used (argmax)
        num_used_classes = torch.unique(p.argmax(dim=-1)).numel()

        loss.backward()
        clip_grad_norm_(model.parameters(), args.clip, 'inf')
        optimizer.step()
        global_bar.set_postfix({
                "epoch": epoch + 1,
                "step": global_steps,
                "MSE": f"{mse.item():5f}"
        })
        global_bar.update(1) 
        with torch.no_grad():
            slot_mass = mask.squeeze(2).sum(dim=(-1, -2))
            max_mass = slot_mass.max(dim=1).values.mean().item()
            if batch % log_interval == 0 and local_rank == 0:
                print('Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F} \t MSE: {:F}'.format(epoch+1, batch, train_epoch_size, loss.item(), mse.item()))
                writer.add_scalar('TRAIN/loss', loss.item(), global_step)
                writer.add_scalar('TRAIN/mse', mse.item(), global_step)
                writer.add_scalar('TRAIN/reg_bck', reg_bck.item(), global_step)
                writer.add_scalar('TRAIN/lr_enc', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('TRAIN/lr_dec', optimizer.param_groups[1]['lr'], global_step)
                writer.add_scalar('TRAIN/max_mass', max_mass, global_step)
                writer.add_scalar('TRAIN/pi_maxprob', pi_maxprob,global_step)
                writer.add_scalar('TRAIN/pi_entropy', pi_entropy, global_step)
                writer.add_scalar('TRAIN/pi_num_used_classes', num_used_classes,global_step)
                writer.add_scalar('TRAIN/KL', klds,global_step)
                writer.add_scalar('TRAIN/Ext_KL', kld_ext,global_step)
                writer.add_scalar('TRAIN/Bck_KL', kld_bck,global_step)
                writer.add_scalar('TRAIN/pi_KL', kld_pi,global_step)
    if (epoch+1)%1==0:
        with torch.no_grad():
            model.eval()

            test_output = {key: [] for key in ['ami_all_attn', 'ari_all_attn', 'ami_obj_attn', 'ari_obj_attn',
                                            'ami_all_mask', 'ari_all_mask', 'ami_obj_mask', 'ari_obj_mask',
                                                ]}
            val_mse = 0.
            pairs_list = []
            count = 0
            for batch, (video, seg) in enumerate(val_loader):
                
                video = video.cuda()
                B, C, H, W = video.size()
                seg = seg.squeeze(1)
                if seg[0, 0, 0] != 0:
                    segment = (seg + 1) % (seg.max() + 1)
                else:
                    segment = seg

                (mse, _, _, attns, masks,pi_c,pi_logist,  kld_ext, kld_bck, kld_pi) = model(video, tau,1)

                p = torch.softmax(pi_logist.detach(), dim=-1) 
                # Mean usage of each class across batch+slots
                p_mean = p.mean(dim=(0, 1))                     # [C]
                # How confident assignments are
                pi_maxprob = p.max(dim=-1).values.mean().item()
                # Entropy (0 = fully collapsed)
                pi_entropy = (-(p.clamp_min(1e-9) * p.clamp_min(1e-9).log())
                            .sum(dim=-1)).mean().item()
                # How many classes are actually used (argmax)
                num_used_classes = torch.unique(p.argmax(dim=-1)).numel()

                attn = torch.nn.functional.interpolate(
                    attns.flatten(end_dim=1), 
                    size=(args.image_size, args.image_size),
                    mode='bilinear'
                )   # [B*K, 1, H, W]
                attns = attn.reshape(*attns.shape[:2], 1, args.image_size, args.image_size )

                mask = torch.nn.functional.interpolate(
                    masks.flatten(end_dim=1), 
                    size=(args.image_size, args.image_size),
                    mode='bilinear'
                )   # [B*K, 1, H, W]
                masks = mask.reshape(*masks.shape[:2], 1, args.image_size, args.image_size )
    
                # calculate the metrics of attention mask 
                attn = attns.reshape(B, args.num_slots, H, W)
                attn = torch.argmax(attn, dim=1)
                ari_ami = compute_ari_ami(segment, attn)
                #iou, f1 = compute_iou_f1(segment, attn)
                for key, val in ari_ami.items():
                    test_output[key + '_attn'].append(val)
                #test_output['iou_attn'].append(iou)
                #test_output['f1_attn'].append(f1)

                # calculate the metrics of reconstruction mask 
                mask = masks.reshape(B, args.num_slots, H, W)
                mask = torch.argmax(mask, dim=1)
                ari_ami = compute_ari_ami(segment, mask)
                #iou, f1 = compute_iou_f1(segment, mask)
                for key, val in ari_ami.items():
                    test_output[key + '_mask'].append(val)
                #test_output['iou_mask'].append(iou)
                #test_output['f1_mask'].append(f1)

                # Calculate the accuracy of identify
                """segs = segment.reshape(B, H, W).cuda()
                mas = masks.reshape(B, args.num_slots, 1, H, W)
                pairs_batch = compute_pairs(video, segs, mas[:, 1:], clss, pi_c)
                pairs_list.append(pairs_batch)
                """
                val_mse += mse.item()
                count+=1
            val_mse /= count

            # id_acc = cluster_acc(args.num_cls, pairs_list)
            test_output = {key: np.array(val).mean(-1) for key, val in test_output.items()}

            val_loss = -val_mse
            if local_rank == 0:
                writer.add_scalar('VAL/loss', val_loss, epoch+1)
                writer.add_scalar('VAL/mse', val_mse, epoch+1)
                #.add_scalar('VAL/acc', id_acc, epoch+1)
                writer.add_scalar('VAL/ari_obj_attn', test_output['ari_obj_attn'], epoch+1)
                writer.add_scalar('VAL/ari_obj_mask', test_output['ari_obj_mask'], epoch+1)

                print('====> Dataset:{}, Model:{},  Epoch: {:3} \t Loss = {:F}'.format(args.data_name, args.model_name, epoch+1, val_loss))
                #print('ACC: {:.4f}'.format(id_acc))
                print('Metrics: ' + ', '.join(f'{key}: {val:.4f}' for key, val in test_output.items()))


            if local_rank == 0:
                attns = torch.softmax(attns/0.001, dim=-4)
                masks = torch.softmax(masks/0.001, dim=-4)
                video_vis = video #imagenet_unnormalize(video).clamp(0, 1)
                attns_vis = video_vis.unsqueeze(1) * attns + (1 - attns)
                masks_vis = video_vis.unsqueeze(1) * masks + (1 - masks)
                seg_vis = torch.cat([video_vis[:,None], attns_vis, video_vis[:,None], masks_vis], dim=1)
                segdis = make_grid(seg_vis[:4].flatten(end_dim=1), nrow=args.num_slots+1, padding=2, pad_value=0.8)
                writer.add_image('VAL_segment/epoch={:03}'.format(epoch + 1), segdis)

                vis_dis, vis_pro = model.module.visual_dis(video, tau)
                visdis = make_grid(vis_dis, nrow=B, padding=2, pad_value=1)
                writer.add_image('VAL_disent/epoch={:03}'.format(epoch+1), visdis)

                vispro = make_grid(vis_pro, nrow=args.num_cls, padding=2, pad_value=1)
                writer.add_image('VAL_protos/epoch={:03}'.format(epoch+1), vispro)
            

            # Save best model
            if val_loss > best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                if local_rank == 0:
                    torch.save(model.module.state_dict(), os.path.join(log_path, 'best_model.pt'))
                if global_step < args.steps and local_rank == 0:
                    torch.save(model.module.state_dict(), os.path.join(log_path, f'best_model_until_{args.steps}_steps.pt'))
            if local_rank == 0:
                writer.add_scalar('VAL/best_loss', best_val_loss, epoch+1)
            checkpoint = {
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            if local_rank == 0:
                torch.save(checkpoint, os.path.join(log_path, 'checkpoint.pt.tar'))
                print('====> Best Loss = {:F} @ Epoch {}'.format(best_val_loss, best_epoch))

if local_rank == 0:
    writer.close()
