/home/ge.polymtl.ca/p123239/.conda/envs/dino/bin/python -m torch.distributed.run \
  --nnodes 1 --nproc-per-node 2 \
  train_dino.py --train_config_file /home/ge.polymtl.ca/p123239/dinov2-training-HF/configs/dino/config384.yaml
