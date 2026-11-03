#!/bin/bash
#SBATCH --job-name=charge-eval-sim
#SBATCH --account=<account_name>
# Same cluster (Warp) as the model job's GPU nodes, so it can reach their vLLM port.
#SBATCH --partition=batch-warp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.log

# CPU-only sim job; the model runs in its own GPU job (llm_server/serve_model.slurm).
# Submit from this folder, overriding any of the variables below, e.g.:
#   sbatch --export=ALL,NET=/path/PA.net.xml,ROUTES=/path/activitygen.rou.xml,DAYS=5,SEED=7 run_charge_eval.slurm

set -euo pipefail

source "$SLURM_SUBMIT_DIR/setup_sim_env.sh"
cd "$SLURM_SUBMIT_DIR/.."   # GreenEVT/Scripts, where charge_eval.py lives

NET=${NET:?set NET to the PA .net.xml}
ROUTES=${ROUTES:?set ROUTES to the PA ActivityGen route file}
DAYS=${DAYS:-1}                       # must match the ActivityGen schedule
CHARGERS=${CHARGERS:-./chargers_additional.xml}
TRACKER=${TRACKER:-}
POLICIES=${POLICIES:-"base risk llm"}
SEED=${SEED:-42}
OUT=${OUT:-$WORK/charge_eval_out/seed_${SEED}_${SLURM_JOB_ID}}
EXTRA_ARGS=${EXTRA_ARGS:-}            # e.g. "--soc-mean 25 --wh-per-m 0.30"
LLM_WAIT_MINUTES=${LLM_WAIT_MINUTES:-60}

if [[ " $POLICIES " == *" llm "* ]]; then
    # The model job only writes its endpoint file once vLLM answers, so wait for one.
    echo "Waiting up to ${LLM_WAIT_MINUTES} min for a model endpoint in $LLM_ENDPOINT_DIR"
    deadline=$(( $(date +%s) + LLM_WAIT_MINUTES * 60 ))
    until python -c "import charge_policies as c; c.endpoint_from_file('$LLM_ENDPOINT_DIR')" 2>/dev/null; do
        (( $(date +%s) < deadline )) || { echo "No model endpoint came up; is serve_model.slurm queued?"; exit 1; }
        sleep 30
    done
fi

python charge_eval.py \
    --net "$NET" --routes "$ROUTES" --chargers "$CHARGERS" --days "$DAYS" \
    ${TRACKER:+--tracker "$TRACKER"} \
    --policies $POLICIES --seed "$SEED" --out "$OUT" \
    --llm-endpoint-file "$LLM_ENDPOINT_DIR" \
    $EXTRA_ARGS
