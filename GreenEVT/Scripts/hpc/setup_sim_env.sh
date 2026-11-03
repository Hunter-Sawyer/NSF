#!/bin/bash
# Sourced at the start of every sim job: source setup_sim_env.sh
# WORK must match llm_server/setup_env.sh so both sides share LLM_ENDPOINT_DIR.

WORK=${WORK:-<path_to_work_folder>}
VENV=$WORK/sim_venv

export PIP_CACHE_DIR=$WORK/pip_cache
export LLM_ENDPOINT_DIR=$WORK/vllm_endpoints

source "$VENV/bin/activate"
# Prevents conflict with spack python paths
unset PYTHONPATH
# SUMO binaries come from the eclipse-sumo wheel, not spack
export SUMO_HOME=$(python -c "import sumo; print(sumo.SUMO_HOME)")
export PATH=$SUMO_HOME/bin:$PATH
