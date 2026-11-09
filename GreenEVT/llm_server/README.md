# LLM server (HPC)

GPU side of the charge decision evaluation: serves one model with vLLM on
one 40 GB A100 (`batch-warp`, routed to gpunode003-012) for
`Scripts/charge_eval.py`. Submit one job per model:

```bash
sbatch --export=ALL,MODEL=Qwen/Qwen3-8B serve_model.slurm
```

The evaluation uses Qwen/Qwen3-4B, Qwen/Qwen3-8B,
meta-llama/Llama-3.2-1B-Instruct, meta-llama/Llama-3.2-3B-Instruct and
meta-llama/Llama-3.1-8B-Instruct (the Llama repos are gated). The script
adds `--quantization awq` only for AWQ checkpoints and picks the tool-call
parser by model family.

Setup, running, success criteria and troubleshooting: see
[`../CHARGE_EVAL.md`](../CHARGE_EVAL.md).
