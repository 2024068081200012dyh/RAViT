# RAViT: Region-Aware Vision-Language Transformer for Efficient Open-Vocabulary Detection

本项研究旨在解决现有 Open-Vocabulary Detection (OVD) 模型中存在的空间-语义粒度不匹配及计算冗余问题。

![](../../Figures/RAViT.png)

---

## 🚀 核心创新点 (Key Highlights)

* **Region-Align Module**: 放弃了传统的固定 Grid Patch 对齐，引入可学习的区域聚合机制，显著提升了目标边界的定位精度（$AP_{75}$ 提升 4.1%）。
* **Objectness-based Sparse Matching**: 引入目标性预测分支，将匹配复杂度从 $\mathcal{O}(N \times V)$ 降低至 $\mathcal{O}(K \times V)$，在保持高召回的同时大幅提升推理速度。
* **High Efficiency**: 在单张 NVIDIA A100 上达到 **15.2 FPS**，相比基础模型 OWL-ViT 实现了近 2 倍的速度提升。

---

## 🛠️ 安装指南 (Installation)

### 1. 环境准备
建议使用 Python 3.9+ 和 PyTorch 2.1+：

```bash
conda env create -f environment.yaml
conda activate ravit
pip install -r requirements.txt