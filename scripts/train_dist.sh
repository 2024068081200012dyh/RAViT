#！/bin/bash
# Usage: bash scripts/train_dist.sh 8 (for 8 GPUs)
GPUS=$1
PORT=${PORT:-29500}

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py --config configs/ravit_vitb16_lvis.yaml --launcher pytorch