import torch
from models.ravit import RAViT
from datasets import build_dataloader

@torch.no_grad()
def evaluate():
    # 此处加载模型、配置文件及验证集进行AP计算
    print("Starting Zero-shot Evaluation on LVIS...")
    # metrics = calculate_metrics(model, val_loader)
    # print(f"AP_r: {metrics['ap_r']}, AP_75: {metrics['ap_75']}")

if __name__ == "__main__":
    evaluate()