import torch

class SparseSimilarityMatcher:
    """
    RAViT 核心创新点 2 & 3: 目标性过滤与稀疏匹配
    """
    def __init__(self, top_k=300):
        self.top_k = top_k

    @torch.no_grad()
    def forward(self, region_features, text_embeds, objectness_scores):
        """
        Args:
            region_features: [B, R, D] 聚合后的区域特征
            text_embeds: [V, D] 文本分类向量 (Vocabulary Size V)
            objectness_scores: [B, R, 1] 目标性分支预测的分数
        """
        B, R, D = region_features.shape
        
        # 1. Sparse Selection: 根据 Objectness 分数选择前 K 个感兴趣区域
        # 这一步避免了全图 Patch 与全词汇表 V 的计算，复杂度从 O(N*V) 降至 O(K*V)
        _, topk_indices = torch.topk(objectness_scores.squeeze(-1), self.top_k, dim=1) # [B, K]
        
        # 收集选中的特征
        batch_indices = torch.arange(B).view(B, 1).expand(-1, self.top_k)
        sparse_region_features = region_features[batch_indices, topk_indices] # [B, K, D]
        
        # 2. Cross-modal Alignment: 计算视觉区域与文本的余弦相似度
        # 归一化以进行对比学习对齐
        sparse_region_features = F.normalize(sparse_region_features, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        # 矩阵乘法得到 Logits: [B, K, V]
        logits = torch.einsum('bkd,vd->bkv', sparse_region_features, text_embeds)
        
        return logits, topk_indices