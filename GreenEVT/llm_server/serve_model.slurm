#!/bin/bash
#SBATCH --job-name=charge-llm-vllm
#SBATCH --account=<account_name>
#SBATCH --nodelist=gpunode[003-010]
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.log

# Serves one model over an OpenAI-compatible API for Scripts/charge_eval.py.
# Submit from this folder:  sbatch serve_model.slurm
# Override the model with:  sbatch --export=ALL,MODEL=<hf_id> serve_model.slurm

set -euo pipefail

source "$SLURM_SUBMIT_DIR/setup_env.sh"

MODEL=${MODEL:-"Qwen/Qwen3-14B-AWQ"}
mkdir -p "$LLM_ENDPOINT_DIR"
ENDPOINT_FILE="$LLM_ENDPOINT_DIR/${SLURM_JOB_ID}.json"

# Pick a free port so several jobs on one node don't collide
PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

export VLLM_USE_FLASHINFER_SAMPLER=0

# --host 0.0.0.0 so sim jobs on other nodes can reach it.
# The client forces a named tool call (charge_decision), which vLLM serves with
# guided decoding; auto tool choice + hermes parser are kept for free-form use.
# Prompts are ~500 tokens with short replies, so 8k context is plenty and leaves
# more GPU memory for batching concurrent requests.
vllm serve "$MODEL" \
    --quantization awq \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --enable-prefix-caching \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --port "$PORT" \
    --host 0.0.0.0 &
VLLM_PID=$!

trap 'rm -f "$ENDPOINT_FILE"' EXIT

until curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; do
    sleep 5
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died during startup"; exit 1; }
done

# Endpoint file is written only once the server answers, so its existence means "ready".
printf '{"job_id": "%s", "host": "%s", "port": %s, "model": "%s"}\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$PORT" "$MODEL" > "$ENDPOINT_FILE"
echo "vLLM ready: $(cat "$ENDPOINT_FILE")"

wait $VLLM_PID
