"""
Convolutional Transformer for EEG decoding with DSTFormer blocks
层级混合架构：前半部分层使用双Mamba，后半部分层使用时间Mamba+空间Attention
82.21% ± 11.82%,79.82%
"""

import argparse
import os
gpus = [1]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import datetime
import sys
import scipy.io

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
import torch.autograd as autograd
from torchvision.models import vgg19

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

import matplotlib.pyplot as plt
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

# Import Mamba
from mamba_ssm import Mamba


# ==================== 新增：渐进式Dropout调度函数 ====================
def progressive_dropout_schedule(epoch, total_epochs, base_dropout=0.5, max_dropout=0.7):
    """
    渐进式Dropout调度函数
    随着训练进度增加dropout率，在训练后期提供更强的正则化
    
    Args:
        epoch: 当前epoch
        total_epochs: 总epoch数
        base_dropout: 初始dropout率
        max_dropout: 最大dropout率
    
    Returns:
        当前epoch应该使用的dropout率
    """
    progress = epoch / total_epochs
    return base_dropout + (max_dropout - base_dropout) * progress


# ==================== 新增：可调节的Dropout模块 ====================
class AdjustableDropout(nn.Module):
    """
    可动态调整dropout率的Dropout模块
    """
    def __init__(self, initial_p=0.5):
        super().__init__()
        self.initial_p = initial_p
        self.dropout = nn.Dropout(initial_p)
    
    def forward(self, x):
        return self.dropout(x)
    
    def update_dropout_rate(self, p):
        """更新dropout率"""
        self.dropout.p = p


# ==================== 新增：标签平滑损失函数 ====================
class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失函数
    通过将one-hot标签转换为软标签来减少过拟合
    """
    def __init__(self, classes=4, smoothing=0.1, dim=-1):
        """
        Args:
            classes: 类别数量
            smoothing: 平滑参数，取值范围[0, 1]，0表示不平滑
            dim: 计算损失的维度
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        """
        Args:
            pred: 模型预测，shape为(batch_size, num_classes)
            target: 真实标签，shape为(batch_size,)
        """
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # 创建平滑后的标签
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# ==================== 新增：频域增强函数 ====================
def frequency_masking(x, num_masks=2, mask_width=20, p=0.5):
    """
    随机遮挡频域成分进行数据增强
    
    Args:
        x: 输入张量，shape为(batch, channels, height, time)
        num_masks: 每个样本应用的掩码数量
        mask_width: 每个掩码的频率宽度
        p: 应用频域掩蔽的概率
    
    Returns:
        增强后的数据
    """
    if random.random() > p:
        return x
    
    # 对时间维度进行FFT
    x_fft = torch.fft.rfft(x, dim=-1)
    
    batch_size = x.shape[0]
    freq_bins = x_fft.shape[-1]
    
    # 对每个batch中的样本应用掩码
    for b in range(batch_size):
        for _ in range(num_masks):
            # 确保不会超出频率范围
            if freq_bins > mask_width:
                f_start = torch.randint(0, freq_bins - mask_width, (1,)).item()
                # 应用掩码（将选定的频率成分置零）
                x_fft[b, :, :, f_start:f_start+mask_width] = 0
    
    # 逆FFT回到时域
    x_masked = torch.fft.irfft(x_fft, n=x.shape[-1], dim=-1)
    
    return x_masked


# ==================== 新增：相对位置编码 ====================
class RelativePositionBias(nn.Module):
    """
    相对位置偏置，用于增强注意力机制的位置感知能力
    """
    def __init__(self, num_heads, max_len=1000):
        super().__init__()
        self.num_heads = num_heads
        # 初始化相对位置偏置表
        self.bias_table = nn.Parameter(torch.zeros(2 * max_len - 1, num_heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)
        
    def forward(self, seq_len):
        # 生成相对位置偏置
        positions = torch.arange(seq_len, device=self.bias_table.device)
        relative_positions = positions[:, None] - positions[None, :]
        relative_positions += seq_len - 1  # 偏移使其为正数
        
        # 限制索引范围
        max_idx = self.bias_table.size(0) - 1
        relative_positions = relative_positions.clamp(0, max_idx)
        
        bias = self.bias_table[relative_positions]
        return bias.permute(2, 0, 1)  # (num_heads, seq_len, seq_len)


# ==================== 新增：局部注意力机制 ====================
class LocalAttention(nn.Module):
    """
    局部注意力机制，只关注邻近的时间步
    """
    def __init__(self, dim, num_heads, window_size=16, dropout=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = AdjustableDropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 创建局部注意力掩码
        mask = torch.ones(N, N, device=x.device)
        for i in range(N):
            start = max(0, i - self.window_size // 2)
            end = min(N, i + self.window_size // 2 + 1)
            mask[i, start:end] = 0
        mask = mask.bool()
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


# ==================== 改进的多头注意力机制 ====================
class EnhancedMultiHeadAttention(nn.Module):
    """
    增强的多头注意力机制，包含相对位置编码和注意力正则化
    """
    def __init__(self, emb_size, num_heads, dropout, use_relative_position=True, 
                 use_attention_regularization=True):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.use_relative_position = use_relative_position
        self.use_attention_regularization = use_attention_regularization
        
        # QKV投影
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        
        # 相对位置偏置
        if self.use_relative_position:
            self.relative_position_bias = RelativePositionBias(num_heads)
        
        self.att_drop = AdjustableDropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
        # 注意力正则化参数
        if self.use_attention_regularization:
            self.attention_temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        B, N, C = x.shape
        
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        
        # 计算注意力分数
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        
        # 添加相对位置偏置
        if self.use_relative_position:
            relative_bias = self.relative_position_bias(N)
            energy = energy + relative_bias.unsqueeze(0)
        
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        
        # 应用温度调节（注意力正则化）
        if self.use_attention_regularization:
            temperature = self.attention_temperature.clamp(min=0.1, max=10.0)
            att = F.softmax(energy / (scaling * temperature), dim=-1)
        else:
            att = F.softmax(energy / scaling, dim=-1)
            
        att = self.att_drop(att)
        
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        
        return out


# ==================== 新增：混合局部-全局注意力模块 ====================
class HybridAttention(nn.Module):
    """
    混合局部和全局注意力机制
    """
    def __init__(self, dim, num_heads=8, window_size=16, dropout=0., 
                 local_global_ratio=0.5):
        super().__init__()
        self.local_global_ratio = local_global_ratio
        
        # 局部注意力
        self.local_attn = LocalAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout
        )
        
        # 全局注意力
        self.global_attn = EnhancedMultiHeadAttention(
            emb_size=dim,
            num_heads=num_heads,
            dropout=dropout,
            use_relative_position=True,
            use_attention_regularization=True
        )
        
        # 融合层
        self.fusion = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # 并行计算局部和全局注意力
        local_out = self.local_attn(x)
        global_out = self.global_attn(x)
        
        # 自适应融合
        concat_out = torch.cat([local_out, global_out], dim=-1)
        fused_out = self.fusion(concat_out)
        
        # 残差连接和层归一化
        out = self.norm(x + fused_out)
        
        return out


# Convolution module with adjustable dropout
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (22, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
        )
        
        # 使用可调节的Dropout
        self.dropout = AdjustableDropout(0.5)

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.dropout(x)
        x = self.projection(x)
        return x


# Universal Mamba block
class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,  
            d_conv=d_conv,
            expand=expand
        )
        self.dropout = AdjustableDropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, seq_len, dim)
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__()
        self.fc1 = nn.Linear(emb_size, int(expansion * emb_size))
        self.act = nn.GELU()
        self.dropout = AdjustableDropout(drop_p)
        self.fc2 = nn.Linear(int(expansion * emb_size), emb_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Modified TransBlock that can use either attention or Mamba
class TransBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., 
                 drop_path=0., num_heads=8, qkv_bias=False, qk_scale=None, 
                 use_layer_scale=True, layer_scale_init_value=1e-5, mode='spatial',
                 mixer_type="attention", use_temporal_similarity=True, 
                 temporal_connection_len=1, neighbour_num=4, n_frames=61,
                 use_mamba=False, use_hybrid_attention=False, window_size=16):
        super().__init__()
        self.mode = mode
        self.use_mamba = use_mamba
        self.use_hybrid_attention = use_hybrid_attention
        
        if use_mamba:
            # Use Mamba for processing
            self.mixer = MambaBlock(dim, d_state=16, d_conv=4, expand=2, dropout=attn_drop)
        elif use_hybrid_attention:
            # 使用混合局部-全局注意力
            self.norm1 = nn.LayerNorm(dim)
            self.mixer = HybridAttention(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                dropout=attn_drop,
                local_global_ratio=0.5
            )
        else:
            # Use enhanced attention
            self.norm1 = nn.LayerNorm(dim)
            self.mixer = EnhancedMultiHeadAttention(
                dim, num_heads, attn_drop,
                use_relative_position=True,
                use_attention_regularization=True
            )
            
        self.drop_path = AdjustableDropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForwardBlock(dim, mlp_ratio, drop)

    def forward(self, x):
        if self.use_mamba:
            # Direct Mamba processing (already includes residual)
            x = self.mixer(x)
        elif self.use_hybrid_attention:
            # 混合注意力已经包含残差连接
            x = self.mixer(x)
        else:
            # Original attention processing
            x = x + self.drop_path(self.mixer(self.norm1(x)))
        
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# Modified DSTFormerBlock with flexible configuration
class DSTFormerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., 
                 drop_path=0., num_heads=8, use_layer_scale=True, qkv_bias=False, 
                 qk_scale=None, layer_scale_init_value=1e-5, use_adaptive_fusion=True, 
                 hierarchical=False, use_temporal_similarity=True, 
                 temporal_connection_len=1, use_tcn=False, graph_only=False, 
                 neighbour_num=4, n_frames=61, use_spatial_mamba=False, 
                 use_temporal_mamba=False, reverse_order=True, 
                 use_hybrid_attention=False, window_size=16):
        super().__init__()
        self.hierarchical = hierarchical
        self.reverse_order = reverse_order
        dim = dim // 2 if hierarchical else dim
        
        # Spatial-Temporal Blocks with configurable Mamba/Attention
        self.att_spatial = TransBlock(dim, mlp_ratio, act_layer, attn_drop, drop, 
                                    drop_path, num_heads, qkv_bias, qk_scale, 
                                    use_layer_scale, layer_scale_init_value, 
                                    mode='spatial', mixer_type="attention",
                                    use_temporal_similarity=use_temporal_similarity,
                                    neighbour_num=neighbour_num, n_frames=n_frames,
                                    use_mamba=use_spatial_mamba,
                                    use_hybrid_attention=use_hybrid_attention,
                                    window_size=window_size)
        
        self.att_temporal = TransBlock(dim, mlp_ratio, act_layer, attn_drop, drop, 
                                     drop_path, num_heads, qkv_bias, qk_scale, 
                                     use_layer_scale, layer_scale_init_value, 
                                     mode='temporal', mixer_type="attention",
                                     use_temporal_similarity=use_temporal_similarity,
                                     neighbour_num=neighbour_num, n_frames=n_frames,
                                     use_mamba=use_temporal_mamba,
                                     use_hybrid_attention=use_hybrid_attention,
                                     window_size=window_size)
        
        # Graph-based Blocks (keep as attention)
        self.graph_spatial = TransBlock(dim, mlp_ratio, act_layer, attn_drop, drop, 
                                      drop_path, num_heads, qkv_bias, qk_scale, 
                                      use_layer_scale, layer_scale_init_value, 
                                      mode='temporal', mixer_type="attention",
                                      use_temporal_similarity=use_temporal_similarity,
                                      temporal_connection_len=temporal_connection_len,
                                      neighbour_num=neighbour_num, n_frames=n_frames,
                                      use_mamba=False,
                                      use_hybrid_attention=False)
        
        self.graph_temporal = TransBlock(dim, mlp_ratio, act_layer, attn_drop, drop, 
                                       drop_path, num_heads, qkv_bias, qk_scale, 
                                       use_layer_scale, layer_scale_init_value, 
                                       mode='spatial', mixer_type='attention',
                                       use_temporal_similarity=use_temporal_similarity,
                                       temporal_connection_len=temporal_connection_len,
                                       neighbour_num=neighbour_num, n_frames=n_frames,
                                       use_mamba=False,
                                       use_hybrid_attention=False)
        
        self.use_adaptive_fusion = use_adaptive_fusion
        if self.use_adaptive_fusion:
            self.fusion = nn.Linear(dim * 2, 2)
            self._init_fusion()
            
    def _init_fusion(self):
        self.fusion.weight.data.fill_(0)
        self.fusion.bias.data.fill_(0.5)
        
    def forward(self, x):
        """
        x: tensor with shape [B, N, C] where N is sequence length
        """
        # Process with configurable order
        if self.reverse_order:
            # Process temporal first, then spatial
            x_attn = self.att_spatial(self.att_temporal(x))
            x_graph = self.graph_spatial(self.graph_temporal(x))
        else:
            # Original order: spatial first, then temporal
            x_attn = self.att_temporal(self.att_spatial(x))
            x_graph = self.graph_temporal(self.graph_spatial(x))
        
        # Adaptive fusion
        if self.use_adaptive_fusion:
            alpha = torch.cat((x_attn, x_graph), dim=-1)
            alpha = self.fusion(alpha)
            alpha = alpha.softmax(dim=-1)
            x = x_attn * alpha[..., 0:1] + x_graph * alpha[..., 1:2]
        else:
            x = (x_attn + x_graph) / 2
            
        return x


# Original TransformerEncoderBlock for comparison
class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__()
        self.attention = nn.Sequential(
            nn.LayerNorm(emb_size),
            EnhancedMultiHeadAttention(emb_size, num_heads, drop_p),
        )
        self.attn_dropout = AdjustableDropout(drop_p)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(emb_size),
            FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
        )
        self.ff_dropout = AdjustableDropout(drop_p)
        
    def forward(self, x):
        x = x + self.attn_dropout(self.attention(x))
        x = x + self.ff_dropout(self.feed_forward(x))
        return x


# Hierarchical Transformer Encoder with mixed architecture
class HierarchicalTransformerEncoder(nn.Module):
    def __init__(self, depth, emb_size, use_dstformer=True, num_heads=8, 
                 hierarchical_ratio=0.5, reverse_order=True, use_hybrid_attention=True,
                 attention_window_size=16):
        super().__init__()
        self.use_dstformer = use_dstformer
        self.depth = depth - 0  # Actual number of DSTFormer blocks
        
        if use_dstformer:
            self.blocks = nn.ModuleList()
            
            # Calculate split point
            split_point = int(self.depth * hierarchical_ratio)
            
            # First half: Both spatial and temporal use Mamba (fast feature extraction)
            for i in range(split_point):
                block = DSTFormerBlock(
                    dim=emb_size,
                    num_heads=num_heads,
                    drop=0.5,
                    attn_drop=0.5,
                    drop_path=0.1 * i / (self.depth - 1),  # Stochastic depth
                    use_adaptive_fusion=True,
                    n_frames=61,
                    use_spatial_mamba=True,   # Mamba for spatial
                    use_temporal_mamba=True,  # Mamba for temporal
                    reverse_order=reverse_order,
                    use_hybrid_attention=False,  # 前半部分使用Mamba
                    window_size=attention_window_size
                )
                self.blocks.append(block)
            
            # Second half: Temporal uses Mamba, Spatial uses Enhanced Attention
            for i in range(split_point, self.depth):
                block = DSTFormerBlock(
                    dim=emb_size,
                    num_heads=num_heads,
                    drop=0.5,
                    attn_drop=0.5,
                    drop_path=0.1 * i / (self.depth - 1),  # Stochastic depth
                    use_adaptive_fusion=True,
                    n_frames=61,
                    use_spatial_mamba=False,  # Enhanced Attention for spatial
                    use_temporal_mamba=True,   # Mamba for temporal
                    reverse_order=reverse_order,
                    use_hybrid_attention=use_hybrid_attention,  # 后半部分使用混合注意力
                    window_size=attention_window_size
                )
                self.blocks.append(block)
                
            print(f"Hierarchical Architecture: {split_point} Mamba blocks + {self.depth - split_point} Hybrid blocks")
            print(f"Using Enhanced Attention with relative position bias and hybrid local-global attention")
        else:
            # Use original Transformer blocks with enhanced attention
            self.blocks = nn.ModuleList([
                TransformerEncoderBlock(emb_size) for _ in range(depth)
            ])
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    
    def update_dropout_rates(self, drop_rate):
        """更新所有块中的dropout率"""
        for block in self.blocks:
            if hasattr(block, 'att_spatial'):
                # DSTFormerBlock
                self._update_transblock_dropout(block.att_spatial, drop_rate)
                self._update_transblock_dropout(block.att_temporal, drop_rate)
                self._update_transblock_dropout(block.graph_spatial, drop_rate)
                self._update_transblock_dropout(block.graph_temporal, drop_rate)
            else:
                # TransformerEncoderBlock
                if hasattr(block.attn_dropout, 'update_dropout_rate'):
                    block.attn_dropout.update_dropout_rate(drop_rate)
                if hasattr(block.ff_dropout, 'update_dropout_rate'):
                    block.ff_dropout.update_dropout_rate(drop_rate)
    
    def _update_transblock_dropout(self, block, drop_rate):
        """更新TransBlock中的dropout率"""
        if hasattr(block, 'mixer'):
            if hasattr(block.mixer, 'att_drop') and hasattr(block.mixer.att_drop, 'update_dropout_rate'):
                block.mixer.att_drop.update_dropout_rate(drop_rate)
            elif hasattr(block.mixer, 'dropout') and hasattr(block.mixer.dropout, 'update_dropout_rate'):
                block.mixer.dropout.update_dropout_rate(drop_rate)
        
        if hasattr(block, 'mlp') and hasattr(block.mlp, 'dropout'):
            block.mlp.dropout.update_dropout_rate(drop_rate)
        
        if hasattr(block, 'drop_path') and hasattr(block.drop_path, 'update_dropout_rate'):
            # 对于drop_path，我们使用较小的值
            block.drop_path.update_dropout_rate(drop_rate * 0.2)


class ClassificationHead(nn.Module):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(2440, 256),
            nn.ELU(),
            AdjustableDropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            AdjustableDropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out
    
    def update_dropout_rates(self, drop_rate):
        """更新分类头中的dropout率"""
        # 第一个dropout使用原始rate
        if hasattr(self.fc[2], 'update_dropout_rate'):
            self.fc[2].update_dropout_rate(drop_rate)
        # 第二个dropout使用较小的rate
        if hasattr(self.fc[5], 'update_dropout_rate'):
            self.fc[5].update_dropout_rate(drop_rate * 0.6)


class ConformerV3(nn.Module):
    def __init__(self, emb_size=40, depth=6, n_classes=4, use_dstformer=True, 
                 hierarchical_ratio=0.5, reverse_order=True, use_hybrid_attention=True,
                 attention_window_size=16, **kwargs):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size)
        self.transformer = HierarchicalTransformerEncoder(
            depth, emb_size, use_dstformer, 
            hierarchical_ratio=hierarchical_ratio,
            reverse_order=reverse_order,
            use_hybrid_attention=use_hybrid_attention,
            attention_window_size=attention_window_size
        )
        self.classification = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        return self.classification(x)
    
    def update_dropout_rates(self, drop_rate):
        """更新整个模型的dropout率"""
        # 更新PatchEmbedding中的dropout
        if hasattr(self.patch_embedding.dropout, 'update_dropout_rate'):
            self.patch_embedding.dropout.update_dropout_rate(drop_rate)
        
        # 更新Transformer中的dropout
        self.transformer.update_dropout_rates(drop_rate)
        
        # 更新分类头中的dropout
        self.classification.update_dropout_rates(drop_rate)


class ExP():
    def __init__(self, nsub, label_smoothing=0.1, use_progressive_dropout=True, 
                 use_onecycle_lr=True, max_lr=0.001, use_mixup=True, mixup_alpha=0.2,
                 use_freq_masking=True, freq_mask_prob=0.5, num_freq_masks=2, 
                 freq_mask_width=20, use_hybrid_attention=True, attention_window_size=16,
                 log_file=None):
        super(ExP, self).__init__()
        self.batch_size = 72
        self.n_epochs = 500
        self.c_dim = 4
        self.lr = 0.0002  # 基础学习率
        self.max_lr = max_lr  # OneCycleLR的最大学习率
        self.b1 = 0.5
        self.b2 = 0.999
        self.dimension = (190, 50)
        self.nSub = nsub
        self.label_smoothing = label_smoothing  # 标签平滑参数
        self.use_progressive_dropout = use_progressive_dropout  # 是否使用渐进式dropout
        self.use_onecycle_lr = use_onecycle_lr  # 是否使用OneCycleLR
        self.use_mixup = use_mixup  # 是否使用MixUp
        self.mixup_alpha = mixup_alpha  # MixUp的alpha参数
        
        # 频域增强参数
        self.use_freq_masking = use_freq_masking  # 是否使用频域掩蔽
        self.freq_mask_prob = freq_mask_prob  # 应用频域掩蔽的概率
        self.num_freq_masks = num_freq_masks  # 掩码数量
        self.freq_mask_width = freq_mask_width  # 掩码宽度
        
        # 注意力机制参数
        self.use_hybrid_attention = use_hybrid_attention  # 是否使用混合注意力
        self.attention_window_size = attention_window_size  # 局部注意力窗口大小

        self.start_epoch = 0
        self.root = './Datasets/Preprocessed_Data/2a/'

        # Use shared log file
        self.log_file = log_file

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        
        # 使用标签平滑损失
        if self.label_smoothing > 0:
            self.criterion_cls = LabelSmoothingLoss(classes=self.c_dim, smoothing=self.label_smoothing).cuda()
            print(f"Using Label Smoothing with smoothing={self.label_smoothing}")
        else:
            self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
            print("Using standard CrossEntropyLoss")

        # Use ConformerV3 with hierarchical architecture and enhanced attention
        self.model = ConformerV3(
            use_dstformer=True, 
            hierarchical_ratio=0.5,  # 50% Mamba blocks, 50% Hybrid blocks
            reverse_order=True,
            use_hybrid_attention=self.use_hybrid_attention,
            attention_window_size=self.attention_window_size
        ).cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()

    def write_log(self, message):
        """Write to shared log file"""
        if self.log_file:
            self.log_file.write(f"[Subject {self.nSub}] {message}")
            self.log_file.flush()  # Ensure immediate write

    # ==================== 新增：MixUp数据增强函数 ====================
    def mixup_data(self, x, y, alpha=0.2):
        """
        MixUp数据增强
        Args:
            x: 输入数据
            y: 标签
            alpha: Beta分布的参数
        Returns:
            mixed_x: 混合后的输入
            y_a, y_b: 原始标签
            lam: 混合系数
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).cuda()

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, pred, y_a, y_b, lam):
        """
        MixUp损失计算
        """
        return lam * self.criterion_cls(pred, y_a) + (1 - lam) * self.criterion_cls(pred, y_b)

    # Segmentation and Reconstruction (S&R) data augmentation
    def interaug(self, timg, label):  
        aug_data = []
        aug_label = []
        
        # Ensure we know the number of classes
        n_classes = 4
        
        for cls4aug in range(n_classes):
            cls_idx = np.where(label == cls4aug)[0]
            if len(cls_idx) == 0:
                continue
                
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]
            
            if tmp_data.shape[0] < 8:
                continue
                
            tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 22, 1000))
            for ri in range(int(self.batch_size / 4)):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * 125:(rj + 1) * 125]

            aug_data.append(tmp_aug_data)
            aug_label.append(np.full(int(self.batch_size / 4), cls4aug))
        
        if len(aug_data) == 0:
            return torch.zeros((0, 1, 22, 1000)).cuda(), torch.zeros(0).long().cuda()
            
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label).cuda()
        aug_label = aug_label.long()
        return aug_data, aug_label

    def get_source_data(self):
        # train data
        self.total_data = scipy.io.loadmat(self.root + 'A0%dT.mat' % self.nSub)
        self.train_data = self.total_data['data']
        self.train_label = self.total_data['label']

        self.train_data = np.transpose(self.train_data, (2, 1, 0))
        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)

        self.allData = self.train_data
        self.allLabel = self.train_label.flatten()
        
        # Check and adjust label range
        min_label = np.min(self.allLabel)
        max_label = np.max(self.allLabel)
        
        if min_label == 1 and max_label == 4:
            self.allLabel = self.allLabel - 1
        elif min_label < 0:
            self.allLabel = self.allLabel + 1

        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]

        # test data
        self.test_tmp = scipy.io.loadmat(self.root + 'A0%dE.mat' % self.nSub)
        self.test_data = self.test_tmp['data']
        self.test_label = self.test_tmp['label']

        self.test_data = np.transpose(self.test_data, (2, 1, 0))
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = self.test_label.flatten()
        
        # Apply same transformation to test labels
        min_label = np.min(self.testLabel)
        max_label = np.max(self.testLabel)
        
        if min_label == 1 and max_label == 4:
            self.testLabel = self.testLabel - 1
        elif min_label < 0:
            self.testLabel = self.testLabel + 1

        # standardize
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std

        return self.allData, self.allLabel, self.testData, self.testLabel

    def train(self):
        img, label, test_data, test_label = self.get_source_data()

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)

        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        
        # ==================== 新增：OneCycleLR学习率调度器 ====================
        if self.use_onecycle_lr:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.max_lr,
                epochs=self.n_epochs,
                steps_per_epoch=len(self.dataloader),
                pct_start=0.05,  # 5%用于预热
                anneal_strategy='cos'
            )
            self.write_log(f"Using OneCycleLR: max_lr={self.max_lr}, pct_start=0.05, anneal_strategy='cos'\n")
        else:
            self.scheduler = None
            self.write_log(f"Using fixed learning rate: {self.lr}\n")

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        # Train the model
        total_step = len(self.dataloader)
        curr_lr = self.lr

        for e in range(self.n_epochs):
            # ==================== 新增：更新dropout率 ====================
            if self.use_progressive_dropout:
                current_dropout = progressive_dropout_schedule(e, self.n_epochs)
                self.model.module.update_dropout_rates(current_dropout)
                
                # 每50个epoch记录一次当前的dropout率
                if e % 50 == 0:
                    self.write_log(f"Epoch {e}: Updated dropout rate to {current_dropout:.4f}\n")
            
            self.model.train()
            epoch_loss = 0
            epoch_train_acc = 0
            batch_count = 0
            current_lr_values = []  # 记录每个batch的学习率
            
            for i, (img, label) in enumerate(self.dataloader):
                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))

                # data augmentation
                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))

                # ==================== 新增：应用频域增强 ====================
                if self.use_freq_masking:
                    img = frequency_masking(img, num_masks=self.num_freq_masks, 
                                          mask_width=self.freq_mask_width, 
                                          p=self.freq_mask_prob)

                # ==================== 新增：应用MixUp ====================
                if self.use_mixup and self.mixup_alpha > 0:
                    mixed_img, label_a, label_b, lam = self.mixup_data(img, label, self.mixup_alpha)
                    tok, outputs = self.model(mixed_img)
                    loss = self.mixup_criterion(outputs, label_a, label_b, lam)
                else:
                    tok, outputs = self.model(img)
                    loss = self.criterion_cls(outputs, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # ==================== 新增：更新学习率（OneCycleLR） ====================
                if self.scheduler is not None:
                    self.scheduler.step()
                    # 记录当前学习率
                    current_lr = self.optimizer.param_groups[0]['lr']
                    current_lr_values.append(current_lr)
                
                # 累积损失和准确率
                epoch_loss += loss.item()
                
                # 对于MixUp，计算准确率时使用原始标签
                if self.use_mixup and self.mixup_alpha > 0:
                    # 使用混合后的预测，但评估时使用主要标签
                    train_pred = torch.max(outputs, 1)[1]
                    train_acc = float((train_pred == label_a).cpu().numpy().astype(int).sum()) / float(label_a.size(0))
                else:
                    train_pred = torch.max(outputs, 1)[1]
                    train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
                
                epoch_train_acc += train_acc
                batch_count += 1

            # test process
            if (e + 1) % 1 == 0:
                self.model.eval()
                with torch.no_grad():
                    Tok, Cls = self.model(test_data)

                    loss_test = self.criterion_cls(Cls, test_label)
                    y_pred = torch.max(Cls, 1)[1]
                    acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                    
                    # 计算平均训练损失和准确率
                    avg_epoch_loss = epoch_loss / batch_count
                    avg_train_acc = epoch_train_acc / batch_count
                    
                    # 计算当前epoch的平均学习率
                    if self.scheduler is not None and current_lr_values:
                        avg_lr = np.mean(current_lr_values)
                        log_message = ('Epoch: %d, LR: %.6f, Train loss: %.6f, Test loss: %.6f, Train accuracy %.6f, Test accuracy: %.6f, Gap: %.6f\n' % 
                                     (e, avg_lr, avg_epoch_loss, loss_test.detach().cpu().numpy(), avg_train_acc, acc, avg_train_acc - acc))
                    else:
                        log_message = ('Epoch: %d, Train loss: %.6f, Test loss: %.6f, Train accuracy %.6f, Test accuracy: %.6f, Gap: %.6f\n' % 
                                     (e, avg_epoch_loss, loss_test.detach().cpu().numpy(), avg_train_acc, acc, avg_train_acc - acc))
                    
                    print(log_message.strip())
                    self.write_log(log_message)
                    
                    # 每50个epoch额外记录学习率信息
                    if e % 50 == 0 and self.scheduler is not None:
                        self.write_log(f"Epoch {e}: Current learning rate range: {min(current_lr_values):.6f} - {max(current_lr_values):.6f}\n")

                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred

        averAcc = averAcc / num
        
        final_message = f'The average accuracy is: {averAcc}\nThe best accuracy is: {bestAcc}\n'
        print(final_message.strip())
        self.write_log(final_message)

        return bestAcc, averAcc, Y_true, Y_pred


def main():
    best = 0
    aver = 0
    
    # Create results directory
    results_dir = "./results/conformer_v3_attention_optimization"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create timestamp for file naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Open shared log files
    log_file_path = os.path.join(results_dir, f"training_log_attention_{timestamp}.txt")
    result_file_path = os.path.join(results_dir, f"final_results_attention_{timestamp}.txt")
    
    log_file = open(log_file_path, "w")
    result_file = open(result_file_path, "w")
    
    # 参数设置
    label_smoothing = 0.1  # 标签平滑参数
    use_progressive_dropout = True  # 是否使用渐进式dropout
    use_onecycle_lr = True  # 是否使用OneCycleLR
    max_lr = 0.001  # OneCycleLR的最大学习率
    use_mixup = True  # 是否使用MixUp
    mixup_alpha = 0.2  # MixUp的alpha参数
    use_freq_masking = True  # 是否使用频域掩蔽
    freq_mask_prob = 0.5  # 应用频域掩蔽的概率
    num_freq_masks = 2  # 掩码数量
    freq_mask_width = 20  # 掩码宽度
    use_hybrid_attention = True  # 是否使用混合注意力
    attention_window_size = 16  # 局部注意力窗口大小
    
    # Write headers
    log_file.write(f"ConformerV3 with Attention Optimization - Training Log\n")
    log_file.write(f"Label Smoothing Parameter: {label_smoothing}\n")
    log_file.write(f"Progressive Dropout: {'Enabled' if use_progressive_dropout else 'Disabled'}\n")
    log_file.write(f"Dropout Schedule: 0.5 → 0.7 (progressive)\n")
    log_file.write(f"OneCycleLR: {'Enabled' if use_onecycle_lr else 'Disabled'}\n")
    if use_onecycle_lr:
        log_file.write(f"  - Max LR: {max_lr}\n")
        log_file.write(f"  - Warmup: 5% of total steps\n")
        log_file.write(f"  - Anneal Strategy: cosine\n")
    log_file.write(f"MixUp: {'Enabled' if use_mixup else 'Disabled'}\n")
    if use_mixup:
        log_file.write(f"  - Alpha: {mixup_alpha}\n")
    log_file.write(f"Frequency Masking: {'Enabled' if use_freq_masking else 'Disabled'}\n")
    if use_freq_masking:
        log_file.write(f"  - Probability: {freq_mask_prob}\n")
        log_file.write(f"  - Number of masks: {num_freq_masks}\n")
        log_file.write(f"  - Mask width: {freq_mask_width}\n")
    log_file.write(f"Hybrid Attention: {'Enabled' if use_hybrid_attention else 'Disabled'}\n")
    if use_hybrid_attention:
        log_file.write(f"  - Local attention window size: {attention_window_size}\n")
        log_file.write(f"  - Using relative position bias\n")
        log_file.write(f"  - Using attention temperature regularization\n")
    log_file.write(f"Configuration: First 50% layers = Mamba (both spatial & temporal)\n")
    log_file.write(f"              Last 50% layers = Hybrid (Temporal=Mamba, Spatial=Enhanced Attention)\n")
    log_file.write(f"Processing Order: Temporal → Spatial (Reversed)\n")
    log_file.write(f"Started at: {datetime.datetime.now()}\n")
    log_file.write("="*80 + "\n\n")
    
    result_file.write(f"ConformerV3 with Attention Optimization - Results Summary\n")
    result_file.write(f"Label Smoothing Parameter: {label_smoothing}\n")
    result_file.write(f"Progressive Dropout: {'Enabled' if use_progressive_dropout else 'Disabled'}\n")
    result_file.write(f"Dropout Schedule: 0.5 → 0.7 (progressive)\n")
    result_file.write(f"OneCycleLR: {'Enabled' if use_onecycle_lr else 'Disabled'}\n")
    if use_onecycle_lr:
        result_file.write(f"  - Max LR: {max_lr}\n")
        result_file.write(f"  - Warmup: 5% of total steps\n")
        result_file.write(f"  - Anneal Strategy: cosine\n")
    result_file.write(f"MixUp: {'Enabled' if use_mixup else 'Disabled'}\n")
    if use_mixup:
        result_file.write(f"  - Alpha: {mixup_alpha}\n")
    result_file.write(f"Frequency Masking: {'Enabled' if use_freq_masking else 'Disabled'}\n")
    if use_freq_masking:
        result_file.write(f"  - Probability: {freq_mask_prob}\n")
        result_file.write(f"  - Number of masks: {num_freq_masks}\n")
        result_file.write(f"  - Mask width: {freq_mask_width}\n")
    result_file.write(f"Hybrid Attention: {'Enabled' if use_hybrid_attention else 'Disabled'}\n")
    if use_hybrid_attention:
        result_file.write(f"  - Local attention window size: {attention_window_size}\n")
        result_file.write(f"  - Using relative position bias\n")
        result_file.write(f"  - Using attention temperature regularization\n")
    result_file.write(f"Configuration: First 50% layers = Mamba (both spatial & temporal)\n")
    result_file.write(f"              Last 50% layers = Hybrid (Temporal=Mamba, Spatial=Enhanced Attention)\n")
    result_file.write(f"Processing Order: Temporal → Spatial (Reversed)\n")
    result_file.write(f"Timestamp: {datetime.datetime.now()}\n")
    result_file.write("="*80 + "\n\n")

    # Store individual subject results
    subject_results = []

    for i in range(9):
        starttime = datetime.datetime.now()

        seed_n = np.random.randint(2021)
        print('Subject %d - Seed is %d' % (i+1, seed_n))
        
        # Set random seeds
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        # Log subject start
        log_file.write(f"\n{'='*60}\n")
        log_file.write(f"Subject {i+1} - Started at {starttime}\n")
        log_file.write(f"Random seed: {seed_n}\n")
        log_file.write(f"{'='*60}\n\n")

        print('Subject %d' % (i+1))
        # 传入所有优化参数
        exp = ExP(i + 1, 
                  label_smoothing=label_smoothing, 
                  use_progressive_dropout=use_progressive_dropout,
                  use_onecycle_lr=use_onecycle_lr,
                  max_lr=max_lr,
                  use_mixup=use_mixup,
                  mixup_alpha=mixup_alpha,
                  use_freq_masking=use_freq_masking,
                  freq_mask_prob=freq_mask_prob,
                  num_freq_masks=num_freq_masks,
                  freq_mask_width=freq_mask_width,
                  use_hybrid_attention=use_hybrid_attention,
                  attention_window_size=attention_window_size,
                  log_file=log_file)

        bestAcc, averAcc, Y_true, Y_pred = exp.train()
        
        endtime = datetime.datetime.now()
        duration = endtime - starttime
        
        # Store results for this subject
        subject_results.append({
            'subject': i+1,
            'seed': seed_n,
            'best_acc': bestAcc,
            'avg_acc': averAcc,
            'duration': duration
        })
        
        # Log completion
        log_file.write(f"\nSubject {i+1} completed in {duration}\n")
        log_file.write(f"Best accuracy: {bestAcc:.6f}\n")
        log_file.write(f"Average accuracy: {averAcc:.6f}\n")
        log_file.write("-"*60 + "\n")
        log_file.flush()

        print('Subject %d duration: %s' % (i+1, duration))
        print('THE BEST ACCURACY IS ' + str(bestAcc))
        
        best = best + bestAcc
        aver = aver + averAcc
        
        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))

    best = best / 9
    aver = aver / 9

    # Write individual subject results to result file
    result_file.write("Individual Subject Results:\n")
    result_file.write("-"*60 + "\n")
    for res in subject_results:
        result_file.write(f"Subject {res['subject']}:\n")
        result_file.write(f"  Random seed: {res['seed']}\n")
        result_file.write(f"  Best accuracy: {res['best_acc']:.6f}\n")
        result_file.write(f"  Average accuracy: {res['avg_acc']:.6f}\n")
        result_file.write(f"  Training duration: {res['duration']}\n")
        result_file.write("-"*40 + "\n")
    
    # Write final summary
    result_file.write("\n" + "="*60 + "\n")
    result_file.write("FINAL SUMMARY:\n")
    result_file.write("="*60 + "\n")
    result_file.write(f"Average Best accuracy across all subjects: {best:.6f}\n")
    result_file.write(f"Average Average accuracy across all subjects: {aver:.6f}\n")
    
    # Calculate standard deviation
    best_accs = [res['best_acc'] for res in subject_results]
    std_best = np.std(best_accs)
    result_file.write(f"Standard deviation of Best accuracy: {std_best:.6f}\n")
    
    # Find best and worst subjects
    best_subject = max(subject_results, key=lambda x: x['best_acc'])
    worst_subject = min(subject_results, key=lambda x: x['best_acc'])
    
    result_file.write(f"\nBest performing subject: Subject {best_subject['subject']} with accuracy {best_subject['best_acc']:.6f}\n")
    result_file.write(f"Worst performing subject: Subject {worst_subject['subject']} with accuracy {worst_subject['best_acc']:.6f}\n")
    
    # Total experiment time
    total_time = sum([res['duration'] for res in subject_results], datetime.timedelta())
    result_file.write(f"\nTotal experiment duration: {total_time}\n")
    result_file.write(f"Experiment completed at: {datetime.datetime.now()}\n")
    
    # 改进对比
    result_file.write("\n" + "="*60 + "\n")
    result_file.write("Improvement Analysis:\n")
    result_file.write("-"*60 + "\n")
    result_file.write(f"Baseline (without enhancements): 80.44%\n")
    result_file.write(f"With Label Smoothing only: ~80.6%\n")
    result_file.write(f"With Label Smoothing + Progressive Dropout: ~80.8%\n")
    result_file.write(f"With LS + PD + OneCycleLR: ~80.63%\n")
    result_file.write(f"With LS + PD + OneCycleLR + MixUp: ~81.6%\n")
    result_file.write(f"With Full Optimization (without attention): ~82.09%\n")
    result_file.write(f"With Full Optimization + Attention Enhancement: {best:.2%}\n")
    result_file.write(f"Total Improvement: {(best - 0.8044)*100:.2f}%\n")
    
    # Close files
    log_file.close()
    result_file.close()
    
    print("\n" + "="*60)
    print(f"All results saved to: {results_dir}")
    print(f"  - Training log: {os.path.basename(log_file_path)}")
    print(f"  - Final results: {os.path.basename(result_file_path)}")
    print(f"\nOptimization Settings:")
    print(f"  - Label Smoothing: {label_smoothing}")
    print(f"  - Progressive Dropout: {'Enabled' if use_progressive_dropout else 'Disabled'}")
    print(f"  - OneCycleLR: {'Enabled' if use_onecycle_lr else 'Disabled'}")
    if use_onecycle_lr:
        print(f"    - Max LR: {max_lr}")
    print(f"  - MixUp: {'Enabled' if use_mixup else 'Disabled'}")
    if use_mixup:
        print(f"    - Alpha: {mixup_alpha}")
    print(f"  - Frequency Masking: {'Enabled' if use_freq_masking else 'Disabled'}")
    if use_freq_masking:
        print(f"    - Probability: {freq_mask_prob}")
        print(f"    - Number of masks: {num_freq_masks}")
        print(f"    - Mask width: {freq_mask_width}")
    print(f"  - Hybrid Attention: {'Enabled' if use_hybrid_attention else 'Disabled'}")
    if use_hybrid_attention:
        print(f"    - Local attention window size: {attention_window_size}")
        print(f"    - Using relative position bias")
        print(f"    - Using attention temperature regularization")
    print(f"\nAverage Best accuracy: {best:.6f} ± {std_best:.6f}")
    print(f"Improvement over baseline: {(best - 0.8044)*100:.2f}%")
    print("="*60)


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))