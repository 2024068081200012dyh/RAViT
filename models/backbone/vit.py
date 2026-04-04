import torch
import torch.nn as nn
import timm

class VisionTransformerAdaptor(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", pretrained=True):
        super().__init__()
        # 使用 timm 库获取标准的 ViT 结构
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.model.embed_dim

    def forward(self, x):
        # 返回 Patch 特征 [B, N_patches, Dim]
        # 去掉 CLS token
        x = self.model.patch_embed(x)
        x = self.model.pos_drop(x + self.model.pos_embed[:, 1:, :])
        for blk in self.model.blocks:
            x = blk(x)
        return x