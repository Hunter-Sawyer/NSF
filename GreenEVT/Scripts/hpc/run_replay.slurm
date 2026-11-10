#!/bin/bash
#SBATCH --job-name=charge-eval-replay
#SBATCH --account=<account_name>
# Same cluster (Warp) as the model jobs, so it can reach their vLLM ports.
#SBATCH --partition=batch-warp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.log

# Replays the decision states logged by one charge_eval run through other
# policies/models (no SUMO). Submit from this folder, e.g.:
#   sbatch --export=ALL,STATES=$WORK/charge_eval_out/seed_42_<job>/base,LLM_MODELS="Qwen/Qwen3-8B" run_replay.slurm

set -euo pipefail

source "$SLURM_SUBMIT_DIR/setup_sim_env.sh"
export PYTHONHASHSEED=0
cd "$SLURM_SUBMIT_DIR/.."   # GreenEVT/Scripts

STATES=${STATES:?set STATES to a charge_eval policy folder (or decisions.jsonl)}
POLICIES=${POLICIES:-"base risk llm"}
LLM_MODELS=${LLM_MODELS:-}
SEED=${SEED:-42}                      # must match the run, for risk's riskiness
OUT=${OUT:-$STATES/replay_${SLURM_JOB_ID}.csv}
EXTRA_ARGS=${EXTRA_ARGS:-}            # e.g. "--deterministic"
LLM_WAIT_MINUTES=${LLM_WAIT_MINUTES:-60}

if [[ " $POLICIES " == *" llm "* ]]; then
    deadline=$(( $(date +%s) + LLM_WAIT_MINUTES * 60 ))
    for model in ${LLM_MODELS:-""}; do
        echo "Waiting up to ${LLM_WAIT_MINUTES} min for ${model:-a model} endpoint in $LLM_ENDPOINT_DIR"
        until python -c "import charge_policies as c; c.endpoint_from_file('$LLM_ENDPOINT_DIR', model='$model' or None)" 2>/dev/null; do
            (( $(date +%s) < deadline )) || { echo "No ${model:-model} endpoint came up; is its serve_model.slurm job queued?"; exit 1; }
            sleep 30
        done
    done
fi

python replay_decisions.py "$STATES" \
    --policies $POLICIES --seed "$SEED" --out "$OUT" \
    --llm-endpoint-file "$LLM_ENDPOINT_DIR" \
    ${LLM_MODELS:+--llm-models $LLM_MODELS} \
    $EXTRA_ARGS
