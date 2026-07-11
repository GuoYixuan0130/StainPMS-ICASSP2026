# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import sys
sys.path.insert(0, './sam2_train/modeling')
import timm
print(timm.__file__)
import copy
import math
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from sam2_train.modeling.fpn import FPN
#from .image_encoder import ImageEncoderViT

class Backbone(nn.Module):
    def __init__(
            self,
            cfg
    ):
        super(Backbone, self).__init__()

        backbone = timm.create_model(
            **cfg.prompter.backbone
        )

        self.backbone = backbone
        self.neck = FPN(
            **cfg.prompter.neck
        )

        new_dict = copy.copy(cfg.prompter.neck)
        new_dict['num_outs'] = 1
        self.neck1 = FPN(
            **new_dict
        )

    def forward(self, images):
        x = self.backbone(images)
        return list(self.neck(x)), self.neck1(x)[0]

class AnchorPoints(nn.Module):
    def __init__(self, space=16):
        super(AnchorPoints, self).__init__()
        self.space = space

    def forward(self, images):
        bs, _, h, w = images.shape
        anchors = np.stack(
            np.meshgrid(
                np.arange(np.ceil(w / self.space)),
                np.arange(np.ceil(h / self.space))),
            -1) * self.space

        origin_coord = np.array([w % self.space or self.space, h % self.space or self.space]) / 2
        anchors += origin_coord

        anchors = torch.from_numpy(anchors).float().to(images.device)
        return anchors.repeat(bs, 1, 1, 1)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, drop=0.1):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList()

        for n, k in zip([input_dim] + h, h):
            self.layers.append(nn.Linear(n, k))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.Dropout(drop))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x
    
class DPAP2PNet(nn.Module):
    """ This is the Proposal-aware P2PNet module that performs cell recognition """

    def __init__(
            self,
            backbone,
            # sam_img_encoder,
            num_levels,
            num_classes,
            dropout=0.1,
            space: int = 16,
            hidden_dim: int = 256,
            with_mask=False,
            enable_quality_head: bool = False,
            quality_head_dropout: float | None = None,
            detach_quality_features: bool = False,
            quantize_quality_features_fp16: bool = False,
            export_quality_features: bool = False,
    ):
        """
            Initializes the model.
        """
        super().__init__()
        self.backbone = backbone
        # self.sam_img_encoder = sam_img_encoder
        self.get_aps = AnchorPoints(space)
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.with_mask = with_mask
        self.enable_quality_head = bool(enable_quality_head)
        self.detach_quality_features = bool(detach_quality_features)
        self.quantize_quality_features_fp16 = bool(quantize_quality_features_fp16)
        self.export_quality_features = bool(export_quality_features)
        self.strides = [2 ** (i + 2) for i in range(self.num_levels)]

        self.deform_layer = MLP(hidden_dim, hidden_dim, 2, 2, drop=dropout)

        self.reg_head = MLP(hidden_dim, hidden_dim, 2, 2, drop=dropout)
        self.cls_head = MLP(hidden_dim, hidden_dim, 2, num_classes + 1, drop=dropout)
        # Built only for PromptCredit so the default architecture and RNG path
        # remain byte-for-byte compatible with historical StainPMS runs.
        if self.enable_quality_head:
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(3407)
                quality_dropout = dropout if quality_head_dropout is None else float(quality_head_dropout)
                self.quality_head = MLP(hidden_dim, hidden_dim, 2, 1, drop=quality_dropout)
                # A uniform low prior makes objectness x quality exactly
                # rank-equivalent to objectness at paired-smoke step 0.
                final_layer = self.quality_head.layers[-1]
                if not isinstance(final_layer, nn.Linear):
                    raise TypeError("PromptCredit quality_head must end in nn.Linear")
                nn.init.zeros_(final_layer.weight)
                nn.init.constant_(final_layer.bias, math.log(0.01 / 0.99))

        self.conv = nn.Conv2d(hidden_dim * num_levels, hidden_dim, kernel_size=3, padding=1)

        self.mask_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SyncBatchNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=1)
        )

    def forward(self,
                images):
        feats,feats1 = self.backbone(images)

        proposals =  self.get_aps(images) 
        embedding = feats
        feats_origin = feats
        feat_sizes = [torch.tensor(feat.shape[:1:-1], dtype=torch.float, device=proposals.device) for feat in feats]

        # DPP deformable point proposals
        grid = (2.0 * proposals / self.strides[0] / feat_sizes[0] - 1.0) 
        roi_features = F.grid_sample(feats[0], grid, mode='bilinear', align_corners=True) 
        deltas2deform = self.deform_layer(roi_features.permute(0, 2, 3, 1))
        deformed_proposals = proposals + deltas2deform

        # MSD  multi-scale decoding
        roi_features = []
        for i in range(self.num_levels):
            grid = (2.0 * deformed_proposals / self.strides[i] / feat_sizes[i] - 1.0)
            roi_features.append(F.grid_sample(feats[i], grid, mode='bilinear', align_corners=True))
        roi_features = torch.cat(roi_features, 1)  

        roi_features = self.conv(roi_features).permute(0, 2, 3, 1)
        deltas2refine = self.reg_head(roi_features)  
        pred_coords = deformed_proposals + deltas2refine

        pred_logits = self.cls_head(roi_features)

        output = {
            'pred_coords': pred_coords.flatten(1, 2),
            'pred_logits': pred_logits.flatten(1, 2),
            'pred_masks': F.interpolate(
                self.mask_head(feats1), size=images.shape[2:], mode='bilinear', align_corners=True)
        }
        if self.enable_quality_head:
            quality_features = roi_features.detach() if self.detach_quality_features else roi_features
            # PromptQ caches FP16 detached features.  Feeding the same quantized
            # values online makes cache-vs-online quality logits exactly
            # comparable without changing any point or decoder computation.
            if self.quantize_quality_features_fp16:
                quality_features = quality_features.to(torch.float16).to(roi_features.dtype)
            output['pred_quality_logits'] = self.quality_head(quality_features).flatten(1, 2).squeeze(-1)
            if self.export_quality_features:
                output['quality_roi_features'] = quality_features.detach()

        return output,feats_origin,embedding,feats

def build_model(
        cfg,
        enable_quality_head: bool = False,
        quality_head_dropout: float | None = None,
        detach_quality_features: bool = False,
        quantize_quality_features_fp16: bool = False,
        export_quality_features: bool = False,
):
    backbone = Backbone(cfg)
    
    model = DPAP2PNet(
        backbone,
        num_levels=cfg.prompter.neck.num_outs,
        num_classes=cfg.data.num_classes,
        dropout=cfg.prompter.dropout,
        space=cfg.prompter.space,
        hidden_dim=cfg.prompter.hidden_dim,
        enable_quality_head=enable_quality_head,
        quality_head_dropout=quality_head_dropout,
        detach_quality_features=detach_quality_features,
        quantize_quality_features_fp16=quantize_quality_features_fp16,
        export_quality_features=export_quality_features,
    )

    return model,backbone
 
 
class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.view(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous().view(b, seq_len, num_heads * head_dim)
        return x

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out
