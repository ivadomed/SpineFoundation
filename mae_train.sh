export PYTORCH_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=1 /home/ge.polymtl.ca/p123239/.conda/envs/SpineFoundation/bin/python -m mae_training.build \
    --config ./mae_training/config_mae_training.json \