/home/ge.polymtl.ca/p123239/.conda/envs/SpineFoundation/bin/python -m mae_training.mae_inference.build \
    --model-params ./model/SpineMAE.json \
    --model-ckpt "./mae_training/ckpt/best.ckpt"\
    --data-params ./data_management/data_param.json \
    --training-params ./mae_training/trainer_param.json \
    --outdir "/home/ge.polymtl.ca/p123239/SpineFoundation/mae_training/mae_inference/out_data" \