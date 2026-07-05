from typing import Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from sam2_train.modeling.sam2_utils import MLP
from sam2_train.utils.misc import get_sdpa_settings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = get_sdpa_settings()
USE_FLASH_ATTN = False
MATH_KERNEL_ON = True
OLD_GPU = True

class MFRB(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()

        self.cross_attn_point_to_sam = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.norm2 = nn.LayerNorm(embedding_dim)
        self.cross_attn_sam_to_point = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

    def forward(
        self, point: Tensor, point_pe: Tensor, sam: Tensor, sam_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Cross attention block, tokens attending to image embedding

        q = point
        if point_pe is not None:
            q = q + point_pe
        k = sam + sam_pe
        attn_out = self.cross_attn_point_to_sam(q=q, k=k, v=sam)
        point = point + attn_out
        point = self.norm1(point)

        # Cross attention block, image embedding attending to tokens
        q = point
        if point_pe is not None:
            q = q + point_pe
        k = sam + sam_pe
        attn_out = self.cross_attn_sam_to_point(q=k, k=q, v=q)
        sam = sam + attn_out
        sam = self.norm2(sam)

        return point, sam
    
class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        with torch.backends.cuda.sdp_kernel(
            enable_flash=USE_FLASH_ATTN,
            # if Flash attention kernel is off, then math kernel needs to be enabled
            enable_math=(OLD_GPU and dropout_p > 0.0) or MATH_KERNEL_ON,
            enable_mem_efficient=OLD_GPU,
        ):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out