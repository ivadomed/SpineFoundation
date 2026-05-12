#!/bin/bash
export CONDA_PREFIX=/home/ge.polymtl.ca/p123239/.conda/envs/FM
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

exec ${CONDA_PREFIX}/bin/python /home/ge.polymtl.ca/p123239/FM/analyze_embeddings.py "$@"
