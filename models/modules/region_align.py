import torch
import torch.nn as nn
import torch.nn.functional as F

class RegionAlignModule(nn.Module):
    """
    RAViT 核心创新点 1: 区域感知聚合模块
    将 ViT 输出的固定 Patch 特征聚合成语义连贯的 Region 特征
    """
    def __init__(self, dim, num_regions=300, num_heads=8):
        super().__init__()
        self.num_regions = num_regions
        # 可学习的区域查询 (Learnable Region Queries)
        self.region_queries = nn.Parameter(torch.randn(1, num_regions, dim))
        
        # 交叉注意力机制：用于从 Patch 中提取 Region 特征
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        # 特征精炼层 (FFN)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, patch_features):
        """
        Args:
            patch_features: [B, N_patch, Dim] 来自 ViT Backbone
        Returns:
            region_features: [B, N_region, Dim] 聚合后的区域特征
        """
        B = patch_features.shape[0]
        queries = self.region_queries.expand(B, -1, -1)
        
        # 1. Cross-Attention: Patch -> Region
        # Region Queries 作为 Q, Patch Features 作为 K, V
        attn_out, _ = self.cross_attn(queries, patch_features, patch_features)
        x = self.norm1(queries + attn_out)
        
        # 2. FFN Refinement
        x = self.norm2(x + self.mlp(x))
        
        return x