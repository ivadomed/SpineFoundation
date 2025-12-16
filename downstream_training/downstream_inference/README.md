Inference utilities for SpineFoundation.

Quick start:
- Provide model params JSON, training params JSON (for image size/resolution), and a checkpoint path.
- Run `python inference/build.py --model-params model/SpineMAE.json --training-params training/trainer_param.json --ckpt training/ckpt/best.ckpt --data-params data_management/data_param.json --outdir inference_outputs`.

This will load the model, apply the same GPU resampling, and write predictions as NIfTI or NumPy arrays under the output folder.