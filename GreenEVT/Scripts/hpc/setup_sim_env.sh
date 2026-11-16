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

# On the Rocky 8 nodes the wheel's sumo binary segfaults (exit 139) when run
# directly, but runs fine when started through the system loader. The sim
# launches "sumo" from PATH, so put a wrapper that does that first on PATH.
# Written via a temp file + rename so parallel jobs never see a partial file.
LDSO=/lib64/ld-linux-x86-64.so.2
if [[ -e $LDSO ]]; then
    mkdir -p "$WORK/sumo_wrap"
    _tmp=$(mktemp "$WORK/sumo_wrap/.sumo.XXXXXX")
    printf '#!/bin/bash\nexec %s "$SUMO_HOME/bin/sumo" "$@"\n' "$LDSO" > "$_tmp"
    chmod 755 "$_tmp" && mv -f "$_tmp" "$WORK/sumo_wrap/sumo"
    unset _tmp
    export PATH=$WORK/sumo_wrap:$PATH
fi
