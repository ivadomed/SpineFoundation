#!/usr/bin/env bash
# Extract CLS + patch_mean embeddings for DINOv2 ViT-Large and run analysis.
# Outputs: analysis_output_dinov2_ax/
#          analysis_output_dinov2/embeddings_models_dinov2-vitl14.npy

CUDA_VISIBLE_DEVICES=0 ./run_analysis.sh \
  --data_dir  /home/ge.polymtl.ca/p123239/data_work/01_extracted_v2 \
  --model     models/dinov2-registers-large \
  --output_dir ./analysis_output_dinov2_ax \
  --cache_dir  ./analysis_output_dinov2 \
  --split      both \
  --orientation axial \
  --datasets   canproco,sct-testing-large,nih-ms-mp2rage,data-multi-subject,dcm-brno,dcm-zurich,sci-colorado,sci-paris,sci-zurich,lumbar-vanderbilt,dcm-zurich-lesions \
  --holdout_datasets sct-testing-large,sci-zurich,dcm-zurich-lesions \
  --no_umap
