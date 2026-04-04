import torch
import yaml
from models.ravit import RAViT
from models.matcher.sparse_matcher import SparseSimilarityMatcher

def main():
    # 加载配置
    with open("configs/ravit_vitb16_lvis.yaml", 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    model = RAViT(config).to(device)
    matcher = SparseSimilarityMatcher(top_k=config['model']['top_k'])
    
    # 假设 text_embeds 是从预训练 CLIP 提取的 LVIS 类别嵌入
    # text_embeds = torch.load("weights/lvis_text_embeds.pt").to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['train']['lr']))

    print("RAViT Training Phase Initiated...")
    # 简化版训练循环
    for epoch in range(config['train']['epochs']):
        model.train()
        # for images, targets in dataloader:
        #    ... 执行前向与损失计算 ...
        print(f"Epoch {epoch} completed.")

if __name__ == "__main__":
    main()