#!/usr/bin/env bash
# Recompute CLS + patch_mean embeddings for the standard dataset set.
# Outputs: analysis_output/embeddings_models_curia_axial.npy
#          analysis_output/patch_mean_models_curia_axial.npy

CUDA_VISIBLE_DEVICES=0 ./run_analysis.sh \
  --data_dir  /home/ge.polymtl.ca/p123239/data_work/01_extracted_v2 \
  --model     models/curia \
  --output_dir ./analysis_output_ax \
  --cache_dir  ./analysis_output \
  --split      both \
  --orientation axial \
  --datasets   canproco,sct-testing-large,nih-ms-mp2rage,data-multi-subject,dcm-brno,dcm-zurich,sci-colorado,sci-paris,sci-zurich,lumbar-vanderbilt,dcm-zurich-lesions \
  --holdout_datasets sct-testing-large,sci-zurich,dcm-zurich-lesions \
  --no_umap
