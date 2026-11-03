# Charge decision evaluation -- HPC runbook

Compares three ways an EV driver decides when, where and how much to charge,
on the same SUMO scenario and the same vehicle population:

| policy | what decides |
|---|---|
| `base` | `routing.plan_ev_route` + 80% minimum when leaving the last charger (current logic) |
| `risk` | same planner; a per-driver riskiness score r ~ N(0.5, 0.2) sets the arrival buffer (30% cautious -> 5% risky) and the minimum SoC leaving the last charger (90% -> 50%) |
| `llm` | one LLM call per trip at departure picks charge / station / target SoC (Qwen3-14B-AWQ served by vLLM) |

Two separate SLURM jobs:

```
llm_server/serve_model.slurm      GPU job (A100, gpunode[003-010]) -- vLLM, Qwen3-14B-AWQ
Scripts/hpc/run_charge_eval.slurm CPU job -- SUMO + charge_eval.py, runs base, risk, llm in turn
```

They find each other through `$WORK/vllm_endpoints/<jobid>.json`
(`{"host", "port", "model"}`), which the model job writes once vLLM answers
and deletes when it exits. The sim job waits (up to 60 min) for one to
appear, so both can be submitted back to back.

## Files

| file | purpose |
|---|---|
| `Scripts/charge_eval.py` | runs each policy in SUMO, collects metrics, writes `summary.csv` |
| `Scripts/charge_policies.py` | the three policies + LLM client (stdlib HTTP only) |
| `Scripts/routing.py` | planner (now takes `charge_nodes` and per-station charge curves) |
| `Scripts/hpc/setup_sim_env.sh`, `requirements.txt` | sim venv (SUMO from the `eclipse-sumo` wheel) |
| `Scripts/hpc/smoke/` | tiny 5x5 grid scenario for checking the pipeline works |
| `llm_server/setup_env.sh`, `requirements.txt` | model venv (vLLM) |

## 1. One-time setup

Edit placeholders:
- `WORK=<path_to_project_root>` in **both** `llm_server/setup_env.sh` and
  `Scripts/hpc/setup_sim_env.sh` -- must be identical (shared endpoint folder).
  This is where venvs, HF cache, endpoint files and results live.
- `--account=<account_name>` in **both** `.slurm` files.

Then from an interactive session (login node is too small). Use plain pip,
don't `spack load`:

```bash
hpcshell --account=<account_name>
cd <repo>/GreenEVT

# model venv (Python 3.10-3.12 for vLLM)
python3.12 -m venv $WORK/llm_venv
source llm_server/setup_env.sh
pip install -r llm_server/requirements.txt
deactivate

# sim venv
python3.12 -m venv $WORK/sim_venv
source Scripts/hpc/setup_sim_env.sh
pip install -r Scripts/hpc/requirements.txt
sumo --version     # should print Eclipse SUMO 1.27.1
```

Optional: pre-download the model so the first job doesn't spend its time on it
(with `llm_server/setup_env.sh` sourced):
`huggingface-cli download Qwen/Qwen3-14B-AWQ`

## 2. Smoke test (does it work?)

Uses the bundled grid with a deliberately tiny battery so charging decisions
actually happen. Takes a few minutes once the model is up.

```bash
cd <repo>/GreenEVT/llm_server
sbatch serve_model.slurm

cd ../Scripts/hpc
sbatch --export=ALL,NET=hpc/smoke/dummy_grid.net.xml,ROUTES=hpc/smoke/routes.xml,CHARGERS=hpc/smoke/chargers.xml,EXTRA_ARGS="--soc-mean 25" run_charge_eval.slurm

squeue -u $USER
```

Paths in `NET`/`ROUTES`/`CHARGERS` are relative to `GreenEVT/Scripts`
(the sim job `cd`s there). Logs: `charge-llm-vllm-<jobid>.log` in
`llm_server/`, `charge-eval-sim-<jobid>.log` in `Scripts/hpc/`.

**It works if** the sim log ends with a table and `Wrote .../summary.csv`, and:
- `ev_trips` = 24 and `completed` = 24 for all three policies
- `trips_charging_pct` > 0 for `risk` and `llm` (charging decisions are executed)
- `llm_fallbacks` = 0, or at least small -- fallbacks mean the model's reply
  was unusable; the reason is in the `llm_fallback` column of `llm/trips.csv`
- `llm_latency_s_mean` is a number, not `nan`

Results: `$WORK/charge_eval_out/seed_42_<jobid>/` -- `summary.csv`, and per
policy `trips.csv` (one row per vehicle; for llm includes the prompt and the
model's decision), `stations.csv`, SUMO's `charging_stations_output.xml`.

For reference, a local run of this exact smoke test against a stub model gave
base / risk / llm `trips_charging_pct` of 8.3 / 70.8 / 75.0; base and risk
should match that on the HPC (same seed), llm will differ with the real model.

A `Could not connect to TraCI server ... Retrying` line at the start of each
policy is normal -- SUMO is still starting up.

## 3. Real runs (Palo Alto)

```bash
sbatch --export=ALL,NET=<PA.net.xml>,ROUTES=<PA routes>,SEED=42 run_charge_eval.slurm
```

- `CHARGERS` defaults to `./chargers_additional.xml` (in `Scripts/`).
- EVs are recognised by vType id `electric_vehicle` (`--ev-type` to change)
  and need the battery device params, as in `Scripts/test_PA_rou.xml`.
- Several seeds can run as parallel sim jobs against one model job.
- `POLICIES="base risk"` runs without the model job at all.

Calibrate after the first run:
- **Energy proxy:** the planner assumes `--wh-per-km 170`; the summary's
  `sim_wh_per_km` is what SUMO actually used (~300-370 on the grid). Re-run with
  `EXTRA_ARGS="--wh-per-km <that value>"` or base will under-plan.
- **Does anyone need to charge?** With 50 kWh batteries at ~50% SoC and short
  urban trips, no policy charges and all three look identical. Lower
  `--soc-mean` (or use longer/chained trips) until `trips_charging_pct` is
  meaningfully above 0.

Other useful `EXTRA_ARGS`: `--soc-sd`, `--risk-mean`, `--risk-sd`,
`--llm-concurrency` (default 16 simultaneous requests), `--sample-every`
(raise for big populations if the sim is slow). `python charge_eval.py -h`
lists everything.

## Troubleshooting

| symptom | likely cause / fix |
|---|---|
| sim: `No model endpoint came up` | model job not running or crashed -- check `squeue` and the vLLM log; `WORK` differs between the two setup scripts |
| vLLM log: CUDA / out-of-memory at startup | lower `--gpu-memory-utilization` or `--max-model-len` in `serve_model.slurm`; confirm the job landed on gpunode003-010 |
| vLLM log: driver/CUDA version error | install a vLLM/torch build matching the node's driver (pin `vllm==<ver>` in `llm_server/requirements.txt`) |
| `llm_fallback` = `request_failed: ... 400` | server rejected the request; most often `tool_choice`/`enable_thinking`. For a non-Qwen3 model add `EXTRA_ARGS="--llm-thinking"`; check the vLLM log for the exact error |
| `llm_fallback` = `invalid_decision` | model picked a station not in its list or a bad target -- inspect `llm_decision` in `trips.csv` |
| sim: `ModuleNotFoundError: traci` / `sumo: not found` | sim venv not active or missing packages -- rerun step 1 for the sim venv |
| `plan_status` = `infeasible` for many trips | starting SoC too low to reach any charger under the energy proxy; check `--wh-per-km` and `--soc-mean` |
| LLM run slow | expected ~1 call per EV; check `llm_latency_s_mean`; raise `--llm-concurrency` if many EVs depart together |

`Scripts/traci_testing.py` is the older single-policy script and currently
crashes on `recompute_total_time_from_path`'s arguments; it isn't used here.
