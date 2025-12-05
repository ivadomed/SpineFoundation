# SpineFoundation
This repo is containing all the code related to the spine foundation model. 


To be able to train the model you need to: 

1. Create a venv using the following command: `python -m venv .env_SpineFoundation`
2. Activate your venv by running: `source .env_SpineFoundation/bin/activate` 
3. Install requirements: `pip install requirements.txt` 
4. Have a data folder containing datasets following the BIDS convention. 


Command to run :

```bash 
python -m training.build \
        --model-params ./model/SpineMAE.json \
        --data-params ./data_management/data_param.json \
        --training-params ./training/trainer_param.json \
        --ddp (optional)
```

Replace GPU, YOURSESSION and YOURENV
Trick : put this command in a .sh file and run the file (bash FILE.sh) to exec the command.