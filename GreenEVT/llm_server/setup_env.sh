#!/bin/bash
# Sourced at the start of every model-serving job: source setup_env.sh
# Set WORK once: a folder under your group storage, e.g. /work/projects/<account>/<username>
# (group folders are listed in the message shown at login; home is too small).

WORK=${WORK:-<path_to_work_folder>}
VENV=$WORK/llm_venv

export HF_HOME=$WORK/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export PIP_CACHE_DIR=$WORK/pip_cache
export TOKENIZERS_PARALLELISM=false

# Where serve_model.slurm advertises {host, port, model}; the sim job reads it.
export LLM_ENDPOINT_DIR=$WORK/vllm_endpoints

source "$VENV/bin/activate"
# Prevents conflict with spack python paths
unset PYTHONPATH
# For the CUDA Toolkit must be called after activating venv
export CUDA_HOME=/opt/ohpc/pub/spack/v0.21.1/opt/spack/linux-rocky8-x86_64_v3/gcc-8.5.0/cuda-11.8.0-rk3sf2nwlkk6ikxsf3u3paj2mzgztcg5
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
