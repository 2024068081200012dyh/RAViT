import torch
import torch.nn as nn
from .backbone.vit import VisionTransformerAdaptor
from .modules.region_align import RegionAlignModule
from .modules.objectness import ObjectnessBranch

class RAViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = VisionTransformerAdaptor(config['model']['backbone'])
        dim = config['model']['embed_dim']
        num_r = config['model']['num_regions']

        self.region_align = RegionAlignModule(dim=dim, num_regions=num_r)
        self.objectness_branch = ObjectnessBranch(dim=dim)
        self.bbox_regressor = nn.Linear(dim, 4) # [x, y, w, h]

    def forward(self, images):
        # 1. 提取 Patch 特征
        patch_feats = self.backbone(images) 
        
        # 2. 聚合为 Region 特征 (创新点 1)
        region_feats = self.region_align(patch_feats)
        
        # 3. 预测目标性分数 (创新点 2)
        obj_scores = self.objectness_branch(region_feats)
        
        # 4. 预测边界框
        bboxes = self.bbox_regressor(region_feats)
        
        return {
            "region_feats": region_feats,
            "obj_scores": obj_scores,
            "bboxes": bboxes
        }