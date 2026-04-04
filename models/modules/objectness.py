import torch.nn as nn

class ObjectnessBranch(nn.Module):
    """
    创新点 2: 目标性预测分支，用于筛选高质量 Region
    """
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x) # [B, R, 1]