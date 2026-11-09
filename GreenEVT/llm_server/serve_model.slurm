#!/bin/bash
#SBATCH --job-name=charge-llm-vllm
#SBATCH --account=<account_name>
# A GPU request on batch-warp is routed to gpu-warp: gpunode[003-012], 40 GB A100s.
#SBATCH --partition=batch-warp
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.log

# Serves one model over an OpenAI-compatible API for Scripts/charge_eval.py.
# Submit one job per model from this folder, e.g.:
#   sbatch --export=ALL,MODEL=Qwen/Qwen3-8B serve_model.slurm
# Models used in the evaluation:
#   Qwen/Qwen3-4B  Qwen/Qwen3-8B
#   meta-llama/Llama-3.2-1B-Instruct  meta-llama/Llama-3.2-3B-Instruct
#   meta-llama/Llama-3.1-8B-Instruct  (Llama repos are gated: see CHARGE_EVAL.md)

set -euo pipefail

source "$SLURM_SUBMIT_DIR/setup_env.sh"

MODEL=${MODEL:?set MODEL to a Hugging Face model id, e.g. --export=ALL,MODEL=Qwen/Qwen3-8B}
mkdir -p "$LLM_ENDPOINT_DIR"

# AWQ checkpoints need the flag; the models above run unquantized (bf16).
QUANT_ARGS=()
[[ "$MODEL" == *AWQ* ]] && QUANT_ARGS=(--quantization awq)

# Parser for free-form ("auto") tool calls. The client forces a named tool
# call, which vLLM serves with guided decoding regardless of the parser.
case "$MODEL" in
    *[Ll]lama*) TOOL_PARSER=llama3_json ;;
    *)          TOOL_PARSER=hermes ;;
esac
ENDPOINT_FILE="$LLM_ENDPOINT_DIR/${SLURM_JOB_ID}.json"

# Pick a free port so several jobs on one node don't collide
PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

export VLLM_USE_FLASHINFER_SAMPLER=0

# --host 0.0.0.0 so sim jobs on other nodes can reach it.
# Prompts are ~500 tokens with short replies, so 8k context is plenty and leaves
# more GPU memory for batching concurrent requests.
vllm serve "$MODEL" \
    ${QUANT_ARGS[@]+"${QUANT_ARGS[@]}"} \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_PARSER" \
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
