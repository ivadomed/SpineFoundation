export PYTHONWARNINGS="ignore::RuntimeWarning"

CUDA_VISIBLE_DEVICES="1" /home/ge.polymtl.ca/p123239/.conda/envs/SpineFoundation/bin/python -m training.build \
     --model-params ./model/defaut_SpineMAE.json \
     --data-params ./training/trainer_default.json \
     --training-params ./training/train_param.json \
