cd /home/ge.polymtl.ca/p123239/SpineFoundation && \
CUDA_VISIBLE_DEVICES=0,1 \
/home/ge.polymtl.ca/p123239/.conda/envs/dino/bin/python -m torch.distributed.run \
  --nnodes 1 --nproc-per-node 2 --master_port 29502 \
  train_dino.py --train_config_file configs/dino/config1024.yaml
