# LLM server (HPC)

GPU side of the charge decision evaluation: serves Qwen3-14B-AWQ with vLLM
on one 40 GB A100 (`batch-warp`, routed to gpunode003-012)
for `Scripts/charge_eval.py --policies llm`. The sim runs as a separate CPU
job from `Scripts/hpc/`.

Setup, running, success criteria and troubleshooting: see
[`../CHARGE_EVAL.md`](../CHARGE_EVAL.md).

Change the model with `sbatch --export=ALL,MODEL=<hf_id> serve_model.slurm`
(drop `--quantization awq` in the script for non-AWQ checkpoints).
