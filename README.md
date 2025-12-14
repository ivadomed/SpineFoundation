# SpineFoundation
This repo is containing all the code related to the spine foundation model. 


To be able to train the model you need to: 

1. Create a venv using the following command: `python -m venv .env_SpineFoundation`
2. Activate your venv by running: `source .env_SpineFoundation/bin/activate` 
3. Install requirements: `pip install requirements.txt` 
4. Have a data folder containing datasets following the BIDS convention. 


Command to run :

```bash 
python -m mae_training.build \
        --config ./mae_training/config.json \
        --ddp (optional)
```
