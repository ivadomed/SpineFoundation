export PYTHONWARNINGS="ignore::RuntimeWarning"

set_slot 1 CUDA_VISIBLE_DEVICES="1" /home/ge.polymtl.ca/p123239/.conda/envs/SpineFoundation/bin/python -m training.build \
     --model-params ./model/SpineMAE.json \
     --data-params ./data_management/data_param.json \
     --training-params ./training/trainer_param.json \
