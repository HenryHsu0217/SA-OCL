from GOLD_Code.utils import *
import pdb
from transformers import ViTFeatureExtractor, ViTModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from torch.distributions.kl import kl_divergence
from torch.distributions.categorical import Categorical
from torchvision.utils import save_image
from torch.distributions.normal import Normal
from GOLD_Code.utils import *

def reparameterize_normal(mu, logvar):
    std = torch.exp(0.5 * logvar)
    noise = torch.randn_like(std)
    return mu + std * noise

def compute_kld_normal(mu, logvar, prior_mu, prior_logvar):
    prior_invvar = torch.exp(-prior_logvar)
    kld = 0.5 * (prior_logvar - logvar + prior_invvar * ((mu - prior_mu).square() + logvar.exp()) - 1)
    return kld.sum(-1)

class DINOEncoder(nn.Module):
    """A wrapper for DINO."""
    def __init__(self, resolution: int, patch_size: int = 8, small_size: bool = True):
        super().__init__()
        self.resolution = resolution
        self.patch_size = patch_size
        self.small_size = small_size
        self.version = 's' if self.small_size else 'b'
        self.dino = ViTModel.from_pretrained(f'GOLD_Code/dino')
        for p in self.dino.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.dino(x).last_hidden_state[:, 1:, :]
        out = out.reshape(out.shape[0], self.resolution, self.resolution, out.shape[-1])
        out = out.permute(0, 3, 1, 2)
        return out

    def train(self, mode=True):
        nn.Module.train(self, mode)
        self.dino.eval()
        return self

class SlotAttentionVideo(nn.Module):
    """GOCL Disentangled Slot Attention: Treats all slots equally."""
    def __init__(self, num_iterations, num_slots, ext_size, int_size, num_cls,
                 input_size, slot_size, mlp_hidden_size, epsilon=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.int_size = int_size
        self.ext_size = ext_size
        self.num_cls = num_cls
        self.slot_size = slot_size
        self.epsilon = epsilon

        self.ext_mu = nn.Parameter(torch.Tensor(1, 1, ext_size))
        self.ext_log_sigma = nn.Parameter(torch.Tensor(1, 1, ext_size))
        nn.init.xavier_uniform_(self.ext_mu)
        nn.init.xavier_uniform_(self.ext_log_sigma)

        self.int_mu = nn.Parameter(torch.Tensor(1, 1, num_cls))
        self.int_log_sigma = nn.Parameter(torch.Tensor(1, 1, num_cls))
        nn.init.xavier_uniform_(self.int_mu)
        nn.init.xavier_uniform_(self.int_log_sigma)

        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_mlp_int = nn.LayerNorm(int_size)
        self.norm_mlp_ext = nn.LayerNorm(ext_size)
        self.norm_protos = nn.LayerNorm(int_size)

        self.project_q = linear(self.slot_size, self.slot_size, bias=False)
        self.project_k = linear(input_size, self.slot_size, bias=False)
        self.project_v = linear(input_size, self.slot_size, bias=False)
        self.project_p = linear(int_size, int_size, bias=False)

        self.gru_int = gru_cell(int_size, int_size)
        self.mlp_int = nn.Sequential(
            linear(int_size, mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_hidden_size, int_size))

        self.gru_ext = gru_cell(ext_size, ext_size)
        self.mlp_ext = nn.Sequential(
            linear(ext_size, mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_hidden_size, ext_size))

    def forward(self, inputs, protos, tau):
        B, num_inputs, input_size = inputs.size()

        slots_ext = inputs.new_empty(B, self.num_slots, self.ext_size).normal_()
        slots_ext = self.ext_mu + torch.exp(self.ext_log_sigma) * slots_ext

        slots_int_init = inputs.new_empty(B, self.num_slots, self.num_cls).normal_()
        logits_pi = self.int_mu + torch.exp(self.int_log_sigma) * slots_int_init

        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs) * (self.slot_size ** -0.5)
        v = self.project_v(inputs)
        protos_norm = self.project_p(self.norm_protos(protos))

        for i in range(self.num_iterations):  
            pi = F.gumbel_softmax(logits_pi, tau=tau, hard=False)
            slots_int = (pi[:,:,:,None] * protos_norm[None,None]).sum(-2)
            slots = torch.cat([slots_int, slots_ext], dim=-1)

            q = self.project_q(slots)
            attn_logits = torch.matmul(k, q.transpose(-1, -2))
            attn_vis = F.softmax(attn_logits, dim=-1)

            attn = (attn_vis + self.epsilon)
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            updates = torch.matmul(attn.transpose(-1, -2), v)

            u_int, u_ext = updates.split([self.int_size, self.ext_size], dim=-1)

            slots_ext = self.gru_ext(u_ext.reshape(-1, self.ext_size), slots_ext.reshape(-1, self.ext_size)).reshape(B, self.num_slots, self.ext_size)
            if i < self.num_iterations - 1:
                slots_ext = slots_ext + self.mlp_ext(self.norm_mlp_ext(slots_ext))

            int_upd = self.gru_int(u_int.reshape(-1, self.int_size), slots_int.reshape(-1, self.int_size)).reshape(B, self.num_slots, self.int_size)
            if i < self.num_iterations - 1:
                int_upd = int_upd + self.mlp_int(self.norm_mlp_int(int_upd))

            logits_pi = torch.matmul(int_upd, protos_norm.transpose(1,0)) * (self.int_size ** -0.5)

        return slots_ext, logits_pi, attn_vis

class CartesianPositionalEmbedding(nn.Module):
    def __init__(self, channels, image_size):
        super().__init__()
        self.projection = conv2d(4, channels, 1)
        self.pe = nn.Parameter(self.build_grid(image_size).unsqueeze(0), requires_grad=False)

    def build_grid(self, side_length):
        coords = torch.linspace(0., 1., side_length + 1)
        coords = 0.5 * (coords[:-1] + coords[1:])
        grid_y, grid_x = torch.meshgrid(coords, coords)
        return torch.stack((grid_x, grid_y, 1 - grid_x, 1 - grid_y), dim=0)

    def forward(self, inputs):
        return inputs + self.projection(self.pe)

class STEVEEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.data_name in ['clevr', 'shop', 'octa']:
            self.cnn = nn.Sequential(
                Conv2dBlock(args.img_channels, args.cnn_hidden_size, 5, 1 if args.image_size == 64 else 2, 2),
                Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
                Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
                conv2d(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
                conv2d(args.cnn_hidden_size, args.cnn_hidden_size, 5, 2, 2),
                conv2d(args.cnn_hidden_size, args.d_dino, 5, 1),
            )
            self.pos = CartesianPositionalEmbedding(args.d_dino, int(args.num_dino_patches**0.5))  
        
        self.layer_norm = nn.LayerNorm(args.d_dino)
        self.mlp = nn.Sequential(
            linear(args.d_dino, args.d_dino, weight_init='kaiming'),
            nn.ReLU(),
            linear(args.d_dino, args.d_dino)
        )
        self.savi = SlotAttentionVideo(
            args.num_iterations, args.num_slots, args.ext_size, args.int_size, args.num_cls,
            args.d_dino, args.slot_size, args.mlp_hidden_size,
        )
        self.protos = nn.Parameter(torch.randn(args.num_cls, args.int_size))
        self.enc_ext = nn.Sequential(
            linear(args.ext_size, args.mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(args.mlp_hidden_size, 2*args.ext_size)
        )

class PatchDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.slot_proj = nn.Linear(args.slot_size, args.d_dino, bias=True)
        nn.init.xavier_uniform_(self.slot_proj.weight)
        nn.init.zeros_(self.slot_proj.bias)
        self.pos_embed = nn.Parameter(torch.randn(1, args.num_dino_patches, args.d_dino) * 0.02)
        self.decoder_obj = build_mlp(args.d_dino, args.d_dino+1)

class OCL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_iterations, self.num_slots = args.num_iterations, args.num_slots
        self.image_size, self.num_dino_patches = args.image_size, args.num_dino_patches
        self.d_dino, self.ext_size, self.int_size, self.num_cls = args.d_dino, args.ext_size, args.int_size, args.num_cls
        self.ocl_encoder = STEVEEncoder(args)
        self.ocl_decoder = PatchDecoder(args)
        self.dino_encoder = DINOEncoder(resolution=28, patch_size=8, small_size=False)
        self.dino_feature_extractor = ViTFeatureExtractor.from_pretrained('GOLD_Code/dino')

        self.register_buffer('prior_logits_pi', torch.full((args.num_cls,), 1/args.num_cls))
        self.register_buffer('prior_ext_mu', torch.zeros([args.ext_size]))
        self.register_buffer('prior_ext_logvar', torch.zeros([args.ext_size]))

    def encode(self, video, tau):
        B = video.size(0)
        with torch.no_grad():
            dino_feat = self.dino_feature_extractor(video*255)
            feat = np.stack(dino_feat['pixel_values'])
            out_dino = self.dino_encoder(torch.tensor(feat).cuda()).detach()
            feat_sce = out_dino
        if self.args.data_name in ['clevr', 'shop', 'octa']:
            feat_sce = self.ocl_encoder.cnn(video)
            feat_sce = self.ocl_encoder.pos(feat_sce)
        H_enc, W_enc = feat_sce.shape[-2:]
        emb_set = feat_sce.permute(0, 2, 3, 1).flatten(1, 2)
        emb_set = self.ocl_encoder.mlp(self.ocl_encoder.layer_norm(emb_set))
        slots_ext_raw, logits_pi, attns = self.ocl_encoder.savi(emb_set, self.ocl_encoder.protos, tau)
        attns = attns.transpose(-1, -2).reshape(B, self.num_slots, 1, H_enc, W_enc)
        pi = F.gumbel_softmax(logits_pi, tau=tau, hard=False)
        slots_int = (pi[:,:,:, None] * self.ocl_encoder.protos[None,None]).sum(-2)
        ext_mu, ext_logvar = self.ocl_encoder.enc_ext(slots_ext_raw).chunk(2,-1)
        slots_ext = reparameterize_normal(ext_mu, ext_logvar)
        return slots_int, slots_ext, pi, attns, out_dino, ext_mu, ext_logvar, logits_pi

    def decode(self, slots_int, slots_ext,use_softmax=True):
        B, K, _ = slots_ext.size()
        slots_all = torch.cat([slots_int, slots_ext], dim=-1)
        slots_all = self.ocl_decoder.slot_proj(slots_all).flatten(0, 1)
        slots_all = slots_all.unsqueeze(1).expand(-1, self.num_dino_patches, -1) + self.ocl_decoder.pos_embed
        
        output_all = self.ocl_decoder.decoder_obj(slots_all).reshape(B, K, self.num_dino_patches, -1)
        decoded_patches, alpha = output_all.split([self.d_dino, 1], dim=-1)

        if use_softmax and K > 1:
            # Competitive masking for reconstruction
            alpha_mask = alpha.softmax(dim=1) 
        else:
            # Independent masking for visualization (Prevents "All White" problem)
            alpha_mask = torch.sigmoid(alpha)

        masks = alpha_mask.transpose(-1, -2).reshape(B, K, 1, int(self.num_dino_patches**0.5), int(self.num_dino_patches**0.5))
        patches_all = decoded_patches * alpha_mask
        return patches_all, masks

    def forward(self, video, tau, beta):
        B = video.size(0)
        slots_int, slots_ext, pi, attns, out_dino, ext_mu, ext_logvar, logits_pi = self.encode(video, tau)
        patches_all, masks = self.decode(slots_int, slots_ext)
        mix_patch = torch.sum(patches_all, dim=1)
        mse = ((out_dino.permute(0, 2, 3, 1).flatten(1, 2).detach() - mix_patch) ** 2).sum() / B
        kld_ext = compute_kld_normal(ext_mu, ext_logvar, self.prior_ext_mu, self.prior_ext_logvar).sum() / B
        kld_pi = kl_divergence(Categorical(logits=logits_pi), Categorical(self.prior_logits_pi).expand(logits_pi.shape[:-1])).sum() / B
        return mse, (kld_ext + kld_pi), attns, masks, pi, kld_ext, kld_pi

    def visual_dis(self, video, tau):
        slots_int, slots_ext, _, _, _, _, _, _ = self.encode(video, tau)
        
        # Slicing individual slots
        int1, ext1 = slots_int[:, :1], slots_ext[:, :1]
        int2, ext2 = slots_int[:, 1:2], slots_ext[:, 1:2]
        
        # use_softmax=False ensures we see the object shape via Sigmoid
        _, mask1 = self.decode(int1, ext1, use_softmax=False)
        _, mask2 = self.decode(int2, ext2, use_softmax=False)
        _, mask_ex12 = self.decode(int1, ext2, use_softmax=False)
        _, mask_ex21 = self.decode(int2, ext1, use_softmax=False)
        
        mask_exc = torch.cat([mask1[:,0], mask_ex12[:,0], mask2[:,0], mask_ex21[:,0]], dim=0)
        mask_exc = mask_exc.reshape(-1, 1, int(self.num_dino_patches**0.5), int(self.num_dino_patches**0.5))

        # Prototype visualization logic
        randn_ext = torch.randn(4, self.ext_size).to(slots_ext.device)
        rad_ext = randn_ext[:,None].expand(-1, self.num_cls, -1).reshape(-1, 1, self.ext_size)
        rad_protos = self.ocl_encoder.protos[None].expand(4, -1, -1).reshape(-1, 1, self.int_size)
        
        _, mask_pro = self.decode(rad_protos, rad_ext, use_softmax=False)
        mask_pro = mask_pro.reshape(-1, 1, int(self.num_dino_patches**0.5), int(self.num_dino_patches**0.5))
        
        return mask_exc, mask_pro