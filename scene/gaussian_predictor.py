import torch
import torch.nn as nn
import torchvision 

import numpy as np
import torch
from torch.nn.functional import silu

from einops import rearrange, repeat

from utils.general_utils import matrix_to_quaternion, quaternion_raw_multiply
from utils.graphics_utils import fov2focal

#######
#new code
import os
import vedo
from vedo import Points, show
import matplotlib.pyplot as plt
import copy
import re
########

# U-Net implementation from EDM
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

#----------------------------------------------------------------------------
# Fully-connected layer.

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.

class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x, N_views_xa=1):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

#----------------------------------------------------------------------------
# Group normalization.

class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x, N_views_xa=1):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x.to(memory_format=torch.channels_last)

#----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

#----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        b, c = x.shape
        x = rearrange(x, 'b c -> (b c)')
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        x = rearrange(x, '(b c) emb_ch -> b (c emb_ch)', b=b)
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.

class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        b, c = x.shape
        x = rearrange(x, 'b c -> (b c)')
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        x = rearrange(x, '(b c) emb_ch -> b (c emb_ch)', b=b)
        return x

class CrossAttentionBlock(torch.nn.Module):
    def __init__(self, num_channels, num_heads = 1, eps=1e-5):
        super().__init__()

        self.num_heads = 1
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)

        self.norm = GroupNorm(num_channels=num_channels, eps=eps)

        self.q_proj = Conv2d(in_channels=num_channels, out_channels=num_channels, kernel=1, **init_attn)
        self.kv_proj = Conv2d(in_channels=num_channels, out_channels=num_channels*2, kernel=1, **init_attn)

        self.out_proj = Conv2d(in_channels=num_channels, out_channels=num_channels, kernel=3, **init_zero)

    def forward(self, q, kv):
        q_proj = self.q_proj(self.norm(q)).reshape(q.shape[0] * self.num_heads, q.shape[1] // self.num_heads, -1)
        k_proj, v_proj = self.kv_proj(self.norm(kv)).reshape(kv.shape[0] * self.num_heads, 
                                                   kv.shape[1] // self.num_heads, 2, -1).unbind(2)
        w = AttentionOp.apply(q_proj, k_proj)
        a = torch.einsum('nqk,nck->ncq', w, v_proj)
        x = self.out_proj(a.reshape(*q.shape)).add_(q)

        return x

#----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.

class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if emb_channels is not None:
            self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb=None, N_views_xa=1):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        if emb is not None:
            params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
            if self.adaptive_scale:
                scale, shift = params.chunk(chunks=2, dim=1)
                x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
            else:
                x = silu(self.norm1(x.add_(params)))

        x = silu(self.norm1(x))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            if N_views_xa != 1:
                B, C, H, W = x.shape
                # (B, C, H, W) -> (B/N, N, C, H, W) -> (B/N, N, H, W, C)
                x = x.reshape(B // N_views_xa, N_views_xa, *x.shape[1:]).permute(0, 1, 3, 4, 2)
                # (B/N, N, H, W, C) -> (B/N, N*H, W, C) -> (B/N, C, N*H, W)
                x = x.reshape(B // N_views_xa, N_views_xa * x.shape[2], *x.shape[3:]).permute(0, 3, 1, 2)
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
            if N_views_xa != 1:
                # (B/N, C, N*H, W) -> (B/N, N*H, W, C)
                x = x.permute(0, 2, 3, 1)
                # (B/N, N*H, W, C) -> (B/N, N, H, W, C) -> (B/N, N, C, H, W)
                x = x.reshape(B // N_views_xa, N_views_xa, H, W, C).permute(0, 1, 4, 2, 3)
                # (B/N, N, C, H, W) -> # (B, C, H, W) 
                x = x.reshape(B, C, H, W)
        return x


#----------------------------------------------------------------------------
# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# available at https://github.com/yang-song/score_sde_pytorch
# taken from EDM repository https://github.com/NVlabs/edm/blob/main/training/networks.py#L372

class SongUNet(nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        emb_dim_in           = 0,            # Input embedding dim.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 4,            # Number of residual blocks per resolution.
        attn_resolutions    = [16],         # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise  = 0,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.label_dropout = label_dropout
        self.emb_dim_in = emb_dim_in
        if emb_dim_in > 0:
            emb_channels = model_channels * channel_mult_emb
        else:
            emb_channels = None
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        # Mapping.
        # self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        # self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        # self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        # self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        if emb_dim_in > 0:
            self.map_layer0 = Linear(in_features=emb_dim_in, out_features=emb_channels, **init)
            self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        if noise_channels > 0:
            self.noise_map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
            self.noise_map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, init_weight=0.2, **init)# init_zero)

    def forward(self, x, film_camera_emb=None, N_views_xa=1):

        emb = None

        if film_camera_emb is not None:
            if self.emb_dim_in != 1:
                film_camera_emb = film_camera_emb.reshape(
                    film_camera_emb.shape[0], 2, -1).flip(1).reshape(*film_camera_emb.shape) # swap sin/cos
            film_camera_emb = silu(self.map_layer0(film_camera_emb))
            film_camera_emb = silu(self.map_layer1(film_camera_emb))
            emb = film_camera_emb

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux, N_views_xa)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux, N_views_xa)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux, N_views_xa)) / np.sqrt(2)
            else:
                x = block(x, emb=emb, N_views_xa=N_views_xa) if isinstance(block, UNetBlock) \
                    else block(x, N_views_xa=N_views_xa)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux, N_views_xa)
            elif 'aux_norm' in name:
                tmp = block(x, N_views_xa)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp), N_views_xa)
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    # skip connection is pixel-aligned which is good for
                    # foreground features
                    # but it's not good for gradient flow and background features
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb=emb, N_views_xa=N_views_xa)
        return aux

# ================== End of implementation taken from EDM ===============
# NVIDIA copyright does not apply to the code below this line
def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    
    if max_val - min_val == 0:
        return torch.zeros_like(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    
    return normalized_tensor

class SingleImageSongUNetPredictor(nn.Module):
    def __init__(self, cfg, out_channels, bias, scale):
        super(SingleImageSongUNetPredictor, self).__init__()
        self.out_channels = out_channels
        self.cfg = cfg
        if cfg.cam_embd.embedding is None:
            in_channels = 3
            emb_dim_in = 0
        else:
            in_channels = 3
            emb_dim_in = 6 * cfg.cam_embd.dimension

        self.encoder = SongUNet(cfg.data.training_resolution, 
                                in_channels, 
                                sum(out_channels),
                                model_channels=cfg.model.base_dim,
                                num_blocks=cfg.model.num_blocks,
                                emb_dim_in=emb_dim_in,
                                channel_mult_noise=0,
                                attn_resolutions=cfg.model.attention_resolutions)
        '''self.encoder2 = SongUNet(cfg.data.training_resolution, 
                                in_channels, 
                                sum(out_channels),
                                model_channels=cfg.model.base_dim,
                                num_blocks=cfg.model.num_blocks,
                                emb_dim_in=emb_dim_in,
                                channel_mult_noise=0,
                                attn_resolutions=cfg.model.attention_resolutions)'''
        self.out = nn.Conv2d(in_channels=sum(out_channels), 
                                 out_channels=sum(out_channels),
                                 kernel_size=1)
        ######
        #new code
        self.out2 = nn.Conv2d(in_channels=sum(out_channels), 
                                 out_channels=sum(out_channels),
                                 kernel_size=1)
        ########
        start_channels = 0
        for out_channel, b, s in zip(out_channels, bias, scale):
            nn.init.xavier_uniform_(
                self.out.weight[start_channels:start_channels+out_channel,
                                :, :, :], s)
            nn.init.constant_(
                self.out.bias[start_channels:start_channels+out_channel], b)
            
            start_channels += out_channel
        self.out2 = copy.deepcopy(self.out)


    def forward(self, x, film_camera_emb=None, N_views_xa=1):
        x1 = self.encoder(x, 
                         film_camera_emb=film_camera_emb,
                         N_views_xa=N_views_xa)
        '''x2 = self.encoder2(x, 
                         film_camera_emb=film_camera_emb,
                         N_views_xa=N_views_xa)'''

        return self.out(x1), self.out2(x1)

def networkCallBack(cfg, name, out_channels, **kwargs):
    if name == "SingleUNet":
        return SingleImageSongUNetPredictor(cfg, out_channels, **kwargs)
    else:
        raise NotImplementedError

class GaussianSplatPredictor(nn.Module):
    def __init__(self, cfg):
        super(GaussianSplatPredictor, self).__init__()
        self.cfg = cfg
        assert cfg.model.network_with_offset or cfg.model.network_without_offset, \
            "Need at least one network"

        if cfg.model.network_with_offset:
            split_dimensions, scale_inits, bias_inits = self.get_splits_and_inits(True, cfg)
            self.network_with_offset = networkCallBack(cfg, 
                                        cfg.model.name,
                                        split_dimensions,
                                        scale = scale_inits,
                                        bias = bias_inits)
            assert not cfg.model.network_without_offset, "Can only have one network"
        if cfg.model.network_without_offset:
            split_dimensions, scale_inits, bias_inits = self.get_splits_and_inits(False, cfg)
            self.network_wo_offset = networkCallBack(cfg, 
                                        cfg.model.name,
                                        split_dimensions,
                                        scale = scale_inits,
                                        bias = bias_inits)
            assert not cfg.model.network_with_offset, "Can only have one network"

        self.init_ray_dirs()

        # Activation functions for different parameters
        self.depth_act = nn.Sigmoid()
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        if self.cfg.model.max_sh_degree > 0:
            self.init_sh_transform_matrices()

        if self.cfg.cam_embd.embedding is not None:
            if self.cfg.cam_embd.encode_embedding is None:
                self.cam_embedding_map = nn.Identity()
            elif self.cfg.cam_embd.encode_embedding == "positional":
                self.cam_embedding_map = PositionalEmbedding(self.cfg.cam_embd.dimension)

    def init_sh_transform_matrices(self):
        v_to_sh_transform = torch.tensor([[ 0, 0,-1],
                                          [-1, 0, 0],
                                          [ 0, 1, 0]], dtype=torch.float32)
        sh_to_v_transform = v_to_sh_transform.transpose(0, 1)
        self.register_buffer('sh_to_v_transform', sh_to_v_transform.unsqueeze(0))
        self.register_buffer('v_to_sh_transform', v_to_sh_transform.unsqueeze(0))

    def init_ray_dirs(self):
        x = torch.linspace(-self.cfg.data.training_resolution // 2 + 0.5, 
                            self.cfg.data.training_resolution // 2 - 0.5, 
                            self.cfg.data.training_resolution) 
        y = torch.linspace( self.cfg.data.training_resolution // 2 - 0.5, 
                           -self.cfg.data.training_resolution // 2 + 0.5, 
                            self.cfg.data.training_resolution)
        if self.cfg.model.inverted_x:
            x = -x
        if self.cfg.model.inverted_y:
            y = -y
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        ones = torch.ones_like(grid_x, dtype=grid_x.dtype)
        ray_dirs = torch.stack([grid_x, grid_y, ones]).unsqueeze(0)

        # for cars and chairs the focal length is fixed across dataset
        # so we can preprocess it
        # for co3d this is done on the fly
        if self.cfg.data.category not in ["hydrants", "teddybears"]:
            ray_dirs[:, :2, ...] /= fov2focal(self.cfg.data.fov * np.pi / 180, 
                                              self.cfg.data.training_resolution)
        self.register_buffer('ray_dirs', ray_dirs)

    def get_splits_and_inits(self, with_offset, cfg):
        # Gets channel split dimensions and last layer initialisation
        split_dimensions = []
        scale_inits = []
        bias_inits = []

        if with_offset:
            split_dimensions = split_dimensions + [1, 3, 1, 3, 4, 3]
            scale_inits = scale_inits + [cfg.model.depth_scale, 
                           cfg.model.xyz_scale, 
                           cfg.model.opacity_scale, 
                           cfg.model.scale_scale,
                           1.0,
                           5.0]
            bias_inits = [cfg.model.depth_bias,
                          cfg.model.xyz_bias, 
                          cfg.model.opacity_bias,
                          np.log(cfg.model.scale_bias),
                          0.0,
                          0.0]
        else:
            split_dimensions = split_dimensions + [1, 1, 3, 4, 3]
            scale_inits = scale_inits + [cfg.model.depth_scale, 
                           cfg.model.opacity_scale, 
                           cfg.model.scale_scale,
                           1.0,
                           5.0]
            bias_inits = bias_inits + [cfg.model.depth_bias,
                          cfg.model.opacity_bias,
                          np.log(cfg.model.scale_bias),
                          0.0,
                          0.0]

        if cfg.model.max_sh_degree != 0:
            sh_num = (self.cfg.model.max_sh_degree + 1) ** 2 - 1
            sh_num_rgb = sh_num * 3
            split_dimensions.append(sh_num_rgb)
            scale_inits.append(0.0)
            bias_inits.append(0.0)

        if with_offset:
            self.split_dimensions_with_offset = split_dimensions
        else:
            self.split_dimensions_without_offset = split_dimensions

        return split_dimensions, scale_inits, bias_inits

    def flatten_vector(self, x):
        # Gets rid of the image dimensions and flattens to a point list
        # B x C x H x W -> B x C x N -> B x N x C
        return x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

    def make_contiguous(self, tensor_dict):
        return {k: v.contiguous() for k, v in tensor_dict.items()}

    def multi_view_union(self, tensor_dict, B, N_view):
        for t_name, t in tensor_dict.items():
            t = t.reshape(B, N_view, *t.shape[1:])
            tensor_dict[t_name] = t.reshape(B, N_view * t.shape[2], *t.shape[3:])
        return tensor_dict

    def get_camera_embeddings(self, cameras):
        # get embedding
        # pass through encoding
        b, n_view = cameras.shape[:2]
        if self.cfg.cam_embd.embedding == "index":
            cam_embedding = torch.arange(n_view, 
                                     dtype=cameras.dtype,
                                     device=cameras.device,
                                     ).unsqueeze(0).expand(b, n_view).unsqueeze(2)
        if self.cfg.cam_embd.embedding == "pose":
            # concatenate origin and z-vector. cameras are in row-major order
            cam_embedding = torch.cat([cameras[:, :, 3, :3], cameras[:, :, 2, :3]], dim=2)

        cam_embedding = rearrange(cam_embedding, 'b n_view c -> (b n_view) c')
        cam_embedding = self.cam_embedding_map(cam_embedding)
        cam_embedding = rearrange(cam_embedding, '(b n_view) c -> b n_view c', b=b, n_view=n_view)

        return cam_embedding

    def transform_SHs(self, shs, source_cameras_to_world):
        # shs: B x N x SH_num x 3
        # source_cameras_to_world: B 4 4
        assert shs.shape[2] == 3, "Can only process shs order 1"
        shs = rearrange(shs, 'b n sh_num rgb -> b (n rgb) sh_num')
        transforms = torch.bmm(
            self.sh_to_v_transform.expand(source_cameras_to_world.shape[0], 3, 3),
            # transpose is because source_cameras_to_world is
            # in row major order 
            source_cameras_to_world[:, :3, :3])
        transforms = torch.bmm(transforms, 
            self.v_to_sh_transform.expand(source_cameras_to_world.shape[0], 3, 3))
        
        shs_transformed = torch.bmm(shs, transforms)
        shs_transformed = rearrange(shs_transformed, 'b (n rgb) sh_num -> b n sh_num rgb', rgb=3)

        return shs_transformed

    def transform_rotations(self, rotations, source_cv2wT_quat):
        """
        Applies a transform that rotates the predicted rotations from 
        camera space to world space.
        Args:
            rotations: predicted in-camera rotation quaternions (B x N x 4)
            source_cameras_to_world: transformation quaternions from 
                camera-to-world matrices transposed(B x 4)
        Retures:
            rotations with appropriately applied transform to world space
        """

        Mq = source_cv2wT_quat.unsqueeze(1).expand(*rotations.shape)

        rotations = quaternion_raw_multiply(Mq, rotations) 
        
        return rotations

    def get_pos_from_network_output(self, depth_network, offset, focals_pixels, const_offset=None):

        # expands ray dirs along the batch dimension
        # adjust ray directions according to fov if not done already
        ray_dirs_xy = self.ray_dirs.expand(depth_network.shape[0], 3, *self.ray_dirs.shape[2:])
        if self.cfg.data.category in ["hydrants", "teddybears"]:
            assert torch.all(focals_pixels > 0)
            ray_dirs_xy = ray_dirs_xy.clone()
            ray_dirs_xy[:, :2, ...] = ray_dirs_xy[:, :2, ...] / focals_pixels.unsqueeze(2).unsqueeze(3)

        # depth and offsets are shaped as (b 3 h w)
        if const_offset is not None:
            depth = self.depth_act(depth_network) * (self.cfg.data.zfar - self.cfg.data.znear) + self.cfg.data.znear + const_offset
        else:
            depth = self.depth_act(depth_network) * (self.cfg.data.zfar - self.cfg.data.znear) + self.cfg.data.znear

        pos = ray_dirs_xy * depth + offset

        return pos ,depth ,offset,ray_dirs_xy * depth
    
    def calculate_network_output(self, x, 
                source_cameras_view_to_world, 
                source_cv2wT_quat=None,
                focals_pixels=None,
                activate_output=True):
        B = x.shape[0]
        N_views = x.shape[1]
        # UNet attention will reshape outputs so that there is cross-view attention
        if self.cfg.model.cross_view_attention:
            N_views_xa = N_views
        else:
            N_views_xa = 1

        if self.cfg.cam_embd.embedding is not None:
            cam_embedding = self.get_camera_embeddings(source_cameras_view_to_world)
            assert self.cfg.cam_embd.method == "film"
            film_camera_emb = cam_embedding.reshape(B*N_views, cam_embedding.shape[2])
        else:
            film_camera_emb = None

        if self.cfg.data.category in ["hydrants", "teddybears"]:
            assert focals_pixels is not None
            focals_pixels = focals_pixels.reshape(B*N_views, *focals_pixels.shape[2:])
        else:
            assert focals_pixels is None, "Unexpected argument for non-co3d dataset"

        x = x.reshape(B*N_views, *x.shape[2:])
        if self.cfg.data.origin_distances:
            const_offset = x[:, 3:, ...]
            x = x[:, :3, ...]
        else:
            const_offset = None

        source_cameras_view_to_world = source_cameras_view_to_world.reshape(B*N_views, *source_cameras_view_to_world.shape[2:])
        x = x.contiguous(memory_format=torch.channels_last)

        if self.cfg.model.network_with_offset:

            split_network_outputs = self.network_with_offset(x,
                                                             film_camera_emb=film_camera_emb,
                                                             N_views_xa=N_views_xa
                                                             )
            return split_network_outputs
        else:
            split_network_outputs = self.network_wo_offset(x, 
                                                           film_camera_emb=film_camera_emb,
                                                           N_views_xa=N_views_xa
                                                           ).split(self.split_dimensions_without_offset, dim=1)
            return split_network_outputs
        

    def forward_callback(self, x, 
                source_cameras_view_to_world, 
                source_cv2wT_quat=None,
                focals_pixels=None,
                activate_output=True,saveing_location = None,index = None,newtwork_output =None):

        B = x.shape[0]
        N_views = x.shape[1]
        # UNet attention will reshape outputs so that there is cross-view attention
        if self.cfg.model.cross_view_attention:
            N_views_xa = N_views
        else:
            N_views_xa = 1

        if self.cfg.cam_embd.embedding is not None:
            cam_embedding = self.get_camera_embeddings(source_cameras_view_to_world)
            assert self.cfg.cam_embd.method == "film"
            film_camera_emb = cam_embedding.reshape(B*N_views, cam_embedding.shape[2])
        else:
            film_camera_emb = None

        if self.cfg.data.category in ["hydrants", "teddybears"]:
            assert focals_pixels is not None
            focals_pixels = focals_pixels.reshape(B*N_views, *focals_pixels.shape[2:])
        else:
            assert focals_pixels is None, "Unexpected argument for non-co3d dataset"

        x = x.reshape(B*N_views, *x.shape[2:])
        if self.cfg.data.origin_distances:
            const_offset = x[:, 3:, ...]
            x = x[:, :3, ...]
        else:
            const_offset = None

        source_cameras_view_to_world = source_cameras_view_to_world.reshape(B*N_views, *source_cameras_view_to_world.shape[2:])
        x = x.contiguous(memory_format=torch.channels_last)

        if self.cfg.model.network_with_offset:

            split_network_outputs = newtwork_output[index]

            split_network_outputs = split_network_outputs.split(self.split_dimensions_with_offset, dim=1)
            depth, offset, opacity, scaling, rotation, features_dc = split_network_outputs[:6]
            

            if self.cfg.model.max_sh_degree > 0:
                features_rest = split_network_outputs[6]

            pos ,depth ,offset,points_before_offset = self.get_pos_from_network_output(depth, offset, focals_pixels, const_offset=const_offset)
        else:
            split_network_outputs = newtwork_output[index]

            depth, opacity, scaling, rotation, features_dc = split_network_outputs[:5]
            if self.cfg.model.max_sh_degree > 0:
                features_rest = split_network_outputs[5]

            pos ,depth ,offset,points_before_offset = self.get_pos_from_network_output(depth, 0.0, focals_pixels, const_offset=const_offset)
        ##########
        #new code
        
        if saveing_location is not None:
            for i in range(0,9):
                torchvision.utils.save_image((features_rest[0,i,:,:]).unsqueeze(0),os.path.join(saveing_location, 'features_rest_{}.png'.format(i)))
            torchvision.utils.save_image(features_dc, os.path.join(saveing_location , 'features_dc.png'))
            torchvision.utils.save_image(opacity,os.path.join(saveing_location, 'opacity.png'))
            torchvision.utils.save_image(self.opacity_activation(opacity),os.path.join(saveing_location, 'opacity_Activation.png'))
            torchvision.utils.save_image(normalize_tensor(opacity),os.path.join(saveing_location, 'opacity_norm.png'))
            torchvision.utils.save_image(depth,os.path.join(saveing_location, 'depth.png'))
            depth1 = depth[0,0,:,:]
            plt.imshow(depth1.cpu(), cmap='hot', interpolation='nearest')
            plt.colorbar()
            np.savez(os.path.join(saveing_location, 'depth_data.npz'), depth1=depth1.cpu().numpy())
            torchvision.utils.save_image(normalize_tensor(depth),os.path.join(saveing_location, 'depth_norm.png'))
            normalized_depth = normalize_tensor(depth).cpu().numpy()
            np.savez(os.path.join(saveing_location, 'depth_norm_data.npz'), depth_norm=normalized_depth)
            torchvision.utils.save_image(pos,os.path.join(saveing_location, 'pos.png'))
            torchvision.utils.save_image((pos[0,0,:,:]).unsqueeze(0),os.path.join(saveing_location, 'pos_x.png'))
            torchvision.utils.save_image((pos[0,1,:,:]).unsqueeze(0),os.path.join(saveing_location, 'pos_y.png'))
            torchvision.utils.save_image((pos[0,2,:,:]).unsqueeze(0),os.path.join(saveing_location, 'pos_z.png'))
            torchvision.utils.save_image(((pos[0,0,:,:]+pos[0,1,:,:]+pos[0,2,:,:])/3).unsqueeze(0),os.path.join(saveing_location, 'pos_mean.png'))
            torchvision.utils.save_image((normalize_tensor(pos[0,0,:,:])).unsqueeze(0),os.path.join(saveing_location, 'pos_world_x_norm.png'))
            torchvision.utils.save_image((normalize_tensor(pos[0,1,:,:])).unsqueeze(0),os.path.join(saveing_location, 'pos_world_y_norm.png'))
            torchvision.utils.save_image((normalize_tensor(pos[0,2,:,:])).unsqueeze(0),os.path.join(saveing_location, 'pos_world_z_norm.png'))
            x = pos[0, 0, :, :].cpu().numpy()  # Transfer to CPU and convert to NumPy
            y = pos[0, 1, :, :].cpu().numpy()
            z = pos[0, 2, :, :].cpu().numpy()
            points = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
            point_cloud = Points(points, r=5)  # 'r' sets the radius of each point
            point_cloud.write(os.path.join(saveing_location,"point_cloud.vtk"))

            x = points_before_offset[0, 0, :, :].cpu().numpy()  # Transfer to CPU and convert to NumPy
            y = points_before_offset[0, 1, :, :].cpu().numpy()
            z = points_before_offset[0, 2, :, :].cpu().numpy()
            points = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
            point_cloud = Points(points, r=5)  # 'r' sets the radius of each point
            point_cloud.write(os.path.join(saveing_location,"point_cloud_before_offset.vtk"))
            
            features_rest1 = features_rest[0].permute(1,2,0).cpu().numpy()
            features_dc1 = features_dc[0].permute(1,2,0).cpu().numpy()
            opacity1 = self.flatten_vector(self.opacity_activation(opacity))[0].cpu().numpy()
            np.save(os.path.join(saveing_location,"Sigma.npy"),features_rest1)
            np.save(os.path.join(saveing_location,"features_dc.npy"),features_dc1)
            np.save(os.path.join(saveing_location,"opacity.npy"),opacity1)
            
        #########

        if self.cfg.model.isotropic:
            scaling_out = torch.cat([scaling[:, :1, ...], scaling[:, :1, ...], scaling[:, :1, ...]], dim=1)
        else:
            scaling_out = scaling

        # Pos prediction is in camera space - compute the positions in the world space
        pos = self.flatten_vector(pos)
        pos = torch.cat([pos, 
                         torch.ones((pos.shape[0], pos.shape[1], 1), device=pos.device, dtype=torch.float32)
                         ], dim=2)
        pos = torch.bmm(pos, source_cameras_view_to_world)
        pos = pos[:, :, :3] / (pos[:, :, 3:] + 1e-10)
        
        out_dict = {
            "xyz": pos, 
            "rotation": self.flatten_vector(self.rotation_activation(rotation)),
            "features_dc": self.flatten_vector(features_dc).unsqueeze(2),
            "features_dc_image": features_dc,
            "offset": offset
                }

        if activate_output:
            out_dict["opacity"] = self.flatten_vector(self.opacity_activation(opacity))
            out_dict["scaling"] = self.flatten_vector(self.scaling_activation(scaling_out))
        else:
            out_dict["opacity"] = self.flatten_vector(opacity)
            out_dict["scaling"] = self.flatten_vector(scaling_out)

        assert source_cv2wT_quat is not None
        source_cv2wT_quat = source_cv2wT_quat.reshape(B*N_views, *source_cv2wT_quat.shape[2:])
        out_dict["rotation"] = self.transform_rotations(out_dict["rotation"], 
                    source_cv2wT_quat=source_cv2wT_quat)

        if self.cfg.model.max_sh_degree > 0:
            features_rest = self.flatten_vector(features_rest)
            # Channel dimension holds SH_num * RGB(3) -> renderer expects split across RGB
            # Split channel dimension B x N x C -> B x N x SH_num x 3
            out_dict["features_rest"] = features_rest.reshape(*features_rest.shape[:2], -1, 3)
            assert self.cfg.model.max_sh_degree == 1 # "Only accepting degree 1"
            out_dict["features_rest"] = self.transform_SHs(out_dict["features_rest"],
                                                           source_cameras_view_to_world)
        else:    
            out_dict["features_rest"] = torch.zeros((out_dict["features_dc"].shape[0], 
                                                     out_dict["features_dc"].shape[1], 
                                                     (self.cfg.model.max_sh_degree + 1) ** 2 - 1,
                                                     3), dtype=out_dict["features_dc"].dtype, device="cuda")

        out_dict = self.multi_view_union(out_dict, B, N_views)
        out_dict = self.make_contiguous(out_dict)
        out_dict["normalized_depth"] = self.flatten_vector(depth)
        out_dict["depth_not_flat"] = depth

        return out_dict

    def forward(self, x, 
                source_cameras_view_to_world, 
                source_cv2wT_quat=None,
                focals_pixels=None,
                activate_output=True,saveing_location = None):
        dic = {}
        calculate_output = self.calculate_network_output(x, 
                source_cameras_view_to_world, 
                source_cv2wT_quat=None,
                focals_pixels=None,
                activate_output=True)
        if saveing_location is not None:
            os.makedirs(os.path.join(saveing_location,"back"), exist_ok=True)
            os.makedirs(os.path.join(saveing_location,"front"), exist_ok=True) 
            dic["front"] = self.forward_callback(x, 
                    source_cameras_view_to_world, 
                    source_cv2wT_quat,
                    focals_pixels,
                    activate_output,
                    os.path.join(saveing_location,"front"),index = 0,newtwork_output = calculate_output)
            dic["back"] = self.forward_callback(x, 
                    source_cameras_view_to_world, 
                    source_cv2wT_quat,
                    focals_pixels,
                    activate_output,
                    os.path.join(saveing_location,"back"),index = 1, newtwork_output = calculate_output)

            saveing_location1 = saveing_location.replace("\\", "/")
            script_content = f"""
            import os
            import numpy as np
            import matplotlib.pyplot as plt
            front_path = os.path.join("{saveing_location1}", 'front')
            back_path = os.path.join("{saveing_location1}", 'back')
            depth_data_front = np.load(os.path.join(front_path, 'depth_data.npz'))
            depth1_front = depth_data_front['depth1'].squeeze()
            norm_data_front = np.load(os.path.join(front_path, 'depth_norm_data.npz'))
            depth_norm_front = norm_data_front['depth_norm'].squeeze()
            depth_data_back = np.load(os.path.join(back_path, 'depth_data.npz'))
            depth1_back = depth_data_back['depth1'].squeeze()
            norm_data_back = np.load(os.path.join(back_path, 'depth_norm_data.npz'))
            depth_norm_back = norm_data_back['depth_norm'].squeeze()
            vmin_depth = min(depth1_front.min(), depth1_back.min())
            vmax_depth = max(depth1_front.max(), depth1_back.max())
            vmin_norm = min(depth_norm_front.min(), depth_norm_back.min())
            vmax_norm = max(depth_norm_front.max(), depth_norm_back.max())
            fig, axs = plt.subplots(2, 2, figsize=(12, 12))
            im1 = axs[0, 0].imshow(depth1_front, cmap='hot', vmin=vmin_depth, vmax=vmax_depth, interpolation='nearest')
            axs[0, 0].set_title('Front Depth Heatmap')
            fig.colorbar(im1, ax=axs[0, 0])
            im2 = axs[0, 1].imshow(depth_norm_front, cmap='hot', vmin=vmin_norm, vmax=vmax_norm, interpolation='nearest')
            axs[0, 1].set_title('Front Normalized Depth Heatmap')
            fig.colorbar(im2, ax=axs[0, 1])
            im3 = axs[1, 0].imshow(depth1_back, cmap='hot', vmin=vmin_depth, vmax=vmax_depth, interpolation='nearest')
            axs[1, 0].set_title('Back Depth Heatmap')
            fig.colorbar(im3, ax=axs[1, 0])
            im4 = axs[1, 1].imshow(depth_norm_back, cmap='hot', vmin=vmin_norm, vmax=vmax_norm, interpolation='nearest')
            axs[1, 1].set_title('Back Normalized Depth Heatmap')
            fig.colorbar(im4, ax=axs[1, 1])
            plt.tight_layout()
            plt.savefig(os.path.join("{saveing_location1}", 'combined_heatmaps.png'), dpi=300)
            plt.show()
            """.replace("    ", "")
            with open(os.path.join(saveing_location,"plot_heatmaps.py"), "w") as file:
                file.write(script_content)


            ######################################################
            saveing_location1 = saveing_location.replace("\\", "/")
            script_content = f"""
            import vedo
            import numpy as np
            import os
            def load_points_with_colors(file_path):
                point_cloud = vedo.load(file_path)
                # Extract points and colors
                points = point_cloud.vertices  # Get the vertices (points)
                colors = point_cloud.pointcolors  # Get the point colors
                if colors.shape[1] == 4:
                    colors = colors[:, :3] / 255.0  # Normalize RGB colors if RGBA
                else:
                    colors = colors / 255.0  # Normalize RGB colors if only RGB
                return points, colors

            def visualize_loaded_points(points_list, colors_list, labels):
                plotter = vedo.Plotter()
                for points, colors, label in zip(points_list, colors_list, labels):
                    point_cloud = vedo.Points(points, r=8)  # Set radius of points
                    if colors.shape[1] == 3:
                        colors_rgba = np.hstack((colors, np.ones((colors.shape[0], 1))))  # Add alpha channel
                        colors_rgba = (colors_rgba * 255).astype(np.uint8)  # Convert to 0-255 range
                    else:
                        colors_rgba = colors
                    point_cloud.pointcolors = colors_rgba  # Set the colors
                    point_cloud.point_size(10)  # Set point size
                    plotter += point_cloud
                plotter.show(interactive=True, bg='white')

            def validate_and_visualize(file_paths):
                points_list = []
                colors_list = []
                labels = []
                for file_path in file_paths:
                    points, colors = load_points_with_colors(file_path)
                    points_list.append(points)
                    colors_list.append(colors)
                    labels.append(os.path.basename(file_path))  # Use file name as label
                visualize_loaded_points(points_list, colors_list, labels)

            if __name__ == "__main__":
                file_paths = [
                    os.path.join("{saveing_location1}", 'front', 'point_cloud_with_colors.vtk'),
                    os.path.join("{saveing_location1}", 'back', 'point_cloud_with_colors.vtk')
                ]
                validate_and_visualize(file_paths)
            """

            with open(os.path.join(saveing_location, "point_cloud_visualization_with_colors.py"), "w") as file:
                file.write(script_content)


            ####################
            

        
            saveing_location1 = saveing_location.replace("\\","/")
            function = """
                def visualize_loaded_points(point_clouds, output_gif="point_visualization.gif", duration=5.0, fps=10):
                    video = vedo.Video(output_gif, duration=duration, fps=fps)
                    plotter = vedo.Plotter()
                    n = 4
                    for angle in range(0, 360, 2):
                        for pc in point_clouds:
                            pc.rotate(n, axis=(1, 0, 0))
                        plotter.show(point_clouds, interactive=False, bg='white')
                        video.add_frame()

                    for angle in range(0, 360, 2):
                        for pc in point_clouds:
                            pc.rotate(n, axis=(0, 1, 0))
                        plotter.show(point_clouds, interactive=False, bg='white')
                        video.add_frame()

                    for angle in range(0, 360, 2):
                        for pc in point_clouds:
                            pc.rotate(n, axis=(1, 1, 0))
                        plotter.show(point_clouds, interactive=False, bg='white')
                        video.add_frame()

                    for angle in range(0, 360, 2):
                        for pc in point_clouds:
                            pc.rotate(n, axis=(1, 0, 1))
                        plotter.show(point_clouds, interactive=False, bg='white')
                        video.add_frame()

                    plotter.close()
                    video.close()
                    print("GIF of point visualization saved successfully!")
                """
            function = function.replace("                ","")
            script_content = f"""
                                    import os
                                    import vedo
                                    import numpy as np
                                    {
                                        function
                                    }
                                    xyz_back = vedo.load(os.path.join("{saveing_location1}/back","point_cloud.vtk"))
                                    xyz_front = vedo.load(os.path.join("{saveing_location1}/front","point_cloud.vtk"))
                                    opacity_back = np.load(os.path.join("{saveing_location1}/back","opacity.npy"))
                                    opacity_front = np.load(os.path.join("{saveing_location1}/front","opacity.npy"))
                                    threshold = 0.1
                                    opacity_back [opacity_back < threshold] = 0
                                    opacity_front [opacity_front < threshold] = 0
                                    opacity_front [opacity_front > threshold] = 1
                                    opacity_back [opacity_back > threshold] = 1
                                    xyz_back_color = np.array([[1,0,0] for _ in range(len(xyz_back.vertices))])
                                    xyz_front_color = np.array([[0,0,1] for _ in range(len(xyz_back.vertices))])
                                    xyz_back.point_size(5)
                                    xyz_front.point_size(5)
                                    colors_rgba = np.hstack((xyz_back_color,opacity_back) )
                                    colors_rgba = colors_rgba * 255
                                    xyz_back.pointcolors = colors_rgba
                                    colors_rgba = np.hstack((xyz_front_color,opacity_front) )
                                    colors_rgba = colors_rgba * 255
                                    xyz_front.pointcolors = colors_rgba
                                    #vedo.show([xyz_back,xyz_front], axes=1)
                                    visualize_loaded_points([xyz_back, xyz_front], output_gif="{saveing_location1}/rotating_points_last_one.gif")
            """.replace("                                    ","")
            with open(os.path.join(saveing_location,"point_cloud.py"), "w") as file:
                file.write(script_content)

            saveing_location1 = saveing_location.replace("\\","/")
            script_content = f"""
                                import os
                                import vedo
                                import numpy as np
                                {
                                    function
                                }
                                xyz_back = vedo.load(os.path.join("{saveing_location1}/back","point_cloud_before_offset.vtk"))
                                xyz_front = vedo.load(os.path.join("{saveing_location1}/front","point_cloud_before_offset.vtk"))
                                opacity_back = np.load(os.path.join("{saveing_location1}/back","opacity.npy"))
                                opacity_front = np.load(os.path.join("{saveing_location1}/front","opacity.npy"))
                                threshold = 0.1
                                opacity_back [opacity_back < threshold] = 0
                                opacity_front [opacity_front < threshold] = 0
                                opacity_front [opacity_front > threshold] = 1
                                opacity_back [opacity_back > threshold] = 1
                                xyz_back_color = np.array([[1,0,0] for _ in range(len(xyz_back.vertices))])
                                xyz_front_color = np.array([[0,0,1] for _ in range(len(xyz_back.vertices))])
                                xyz_back.point_size(5)
                                xyz_front.point_size(5)
                                colors_rgba = np.hstack((xyz_back_color,opacity_back) )
                                colors_rgba = colors_rgba * 255
                                xyz_back.pointcolors = colors_rgba
                                colors_rgba = np.hstack((xyz_front_color,opacity_front) )
                                colors_rgba = colors_rgba * 255
                                xyz_front.pointcolors = colors_rgba
                                #vedo.show([xyz_back,xyz_front], axes=1)
                                visualize_loaded_points([xyz_back, xyz_front], output_gif="rotating_points_last_one.gif")
            """.replace("                                ","")
            with open(os.path.join(saveing_location,"point_cloud_before_offset.py"), "w") as file:
                file.write(script_content)
        else: 
            dic["front"] = self.forward_callback(x, 
                    source_cameras_view_to_world, 
                    source_cv2wT_quat,
                    focals_pixels,
                    activate_output,
                    saveing_location,index = 0,newtwork_output = calculate_output)
            dic["back"] = self.forward_callback(x, 
                    source_cameras_view_to_world, 
                    source_cv2wT_quat,
                    focals_pixels,
                    activate_output,
                    saveing_location,index = 1, newtwork_output = calculate_output)
            normalized_depth_front = dic["front"]["normalized_depth"]
            normalized_depth_back = dic["back"]["normalized_depth"]
            dic["front"]["normalized_depth"], dic["back"]["normalized_depth"]= torch.split(torch.cat([normalized_depth_front,normalized_depth_back],dim = 1),[4096,4096],dim=1)

            
        return dic