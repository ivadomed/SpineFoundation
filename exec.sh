CUDA_VISIBLE_DEVICES="YOURGPU" /home/ge.polymtl.ca/YOURSESSION/.conda/envs/YOURENV/bin/python -m training.build \
        --model-params ./model/SpineMAE.json \
        --data-params ./data_management/data_param.json \
        --training-params ./training/trainer_param.json \