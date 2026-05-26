#!/usr/bin/env bash
# Extract CLS + patch_mean embeddings for DINOv3 ViT-Large and run analysis.
# Outputs: analysis_output_dinov3/embeddings_models_dinov3-vitl16.npy
#          analysis_output_dinov3/patch_mean_models_dinov3-vitl16.npy

CUDA_VISIBLE_DEVICES=1 ./run_analysis.sh \
  --data_dir  /home/ge.polymtl.ca/p123239/data_work/01_extracted_v2 \
  --model     models/dinov3-vitl16 \
  --output_dir ./analysis_output_dinov3_ax \
  --cache_dir  ./analysis_output_dinov3 \
  --split      both \
  --orientation axial \
  --datasets   canproco,sct-testing-large,nih-ms-mp2rage,data-multi-subject,dcm-brno,dcm-zurich,sci-colorado,sci-paris,sci-zurich,lumbar-vanderbilt,dcm-zurich-lesions \
  --holdout_datasets sct-testing-large,sci-zurich,dcm-zurich-lesions \
  --no_umap
