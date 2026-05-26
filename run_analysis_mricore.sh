#!/usr/bin/env bash
# Extract embeddings for MRI-CORE (SAM ViT-B, 256-dim) and run analysis.
# Outputs: analysis_output_mricore_ax/
#          analysis_output_mricore/embeddings_models_mricore.npy

CUDA_VISIBLE_DEVICES=1 ./run_analysis.sh \
  --data_dir  /home/ge.polymtl.ca/p123239/data_work/01_extracted_v2 \
  --model     models/mricore \
  --output_dir ./analysis_output_mricore_ax \
  --cache_dir  ./analysis_output_mricore \
  --split      both \
  --orientation axial \
  --datasets   canproco,sct-testing-large,nih-ms-mp2rage,data-multi-subject,dcm-brno,dcm-zurich,sci-colorado,sci-paris,sci-zurich,lumbar-vanderbilt,dcm-zurich-lesions \
  --holdout_datasets sct-testing-large,sci-zurich,dcm-zurich-lesions \
  --no_umap \
  --batch_size 64
