This README describes how to train the model. # SpineFoundation
This repo is containing all the code related to the spine foundation model. 


To be able to train the model you need to: 

1. Create a venv using the following command: `python -m venv .env_SpineFoundation`
2. Activate your venv by running: `source .env_SpineFoundation/bin/activate` 
3. Install requirements: `pip install requirements.txt` 
4. Have a data folder containing datasets following the BIDS convention. 


Command to run :

```bash 
/home/ge.polymtl.ca/p123239/.conda/envs/SpineFoundation/bin/python -m mae_training.mae_inference.build \
    --model-params ./model/SpineMAE.json \
    --model-ckpt "./mae_training/ckpt/best.ckpt"\
    --data-params ./data_management/data_param.json \
    --training-params ./mae_training/trainer_param.json \
    --outdir "/home/ge.polymtl.ca/p123239/SpineFoundation/mae_training/mae_inference/out_data" \
```
