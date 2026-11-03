# Charge decision evaluation -- HPC runbook

Compares three ways an EV driver decides when, where and how much to charge,
inside the team's multi-day ActivityGen sim (`Scripts/traci_testing.py`):
batteries carry over between a household's legs, cars charge at home
overnight, and at the start of every EV leg one decision is made -- drive
on, or detour to a public charger and charge to a target. Only that
decision changes between policies:

| policy | what decides |
|---|---|
| `base` | the sim's own `choose_least_impact_charger`: charge only if the leg can't be finished with a 10% reserve; pick the charger with the least added time; charge to 90% |
| `risk` | the same rule, but each household gets a riskiness score r ~ N(0.5, 0.2) that sets its reserve (30% cautious -> 5% risky) and charge target (90% -> 50%) |
| `llm` | Qwen3-14B-AWQ (vLLM) sees the leg -- time of day, SoC, whether it ends at home, and the same charger options -- and picks charge / station / target SoC |

All three get the same households, first-day battery levels (seeded per
household) and SUMO seed.

Two separate SLURM jobs:

```
llm_server/serve_model.slurm      GPU job (batch-warp, 1x 40 GB A100) -- vLLM, Qwen3-14B-AWQ
Scripts/hpc/run_charge_eval.slurm CPU job (batch-warp) -- SUMO + charge_eval.py, runs base, risk, llm in turn
```

They find each other through `$WORK/vllm_endpoints/<jobid>.json`
(`{"host", "port", "model"}`), which the model job writes once vLLM answers
and deletes when it exits. The sim job waits (up to 60 min) for one to
appear, so both can be submitted back to back.

## Files

| file | purpose |
|---|---|
| `Scripts/traci_testing.py` | the team's multi-day sim; takes an optional `charge_policy` and `trip_log` (default behavior unchanged) |
| `Scripts/charge_eval.py` | runs the sim once per policy, writes per-leg `trips.csv` and a side-by-side `summary.csv` |
| `Scripts/charge_policies.py` | the three policies + LLM client (stdlib HTTP only) |
| `Scripts/hpc/setup_sim_env.sh`, `requirements.txt` | sim venv (SUMO from the `eclipse-sumo` wheel) |
| `Scripts/hpc/smoke/` | tiny 5x5 grid, 12 households commuting for 2 days, for checking the pipeline works |
| `llm_server/setup_env.sh`, `requirements.txt` | model venv (vLLM) |

## 1. One-time setup

**Find your account and storage folder.** Both are printed at the bottom of
the message shown when you log in to `login.hpc.tntech.edu`:

```
Job submission accounts you have access to:
 <account> ...
Group storage folders you have access to:
 /work/projects/<account> ...
```

(`sacctmgr show association format=account%20,user%20` also lists accounts.)
Use the account for this NSF project; if there are several and it's unclear
which, ask the PI. Home directories are very small, so `WORK` should be a
folder in the group storage, e.g. `/work/projects/<account>/<username>`. It
holds the venvs, the model download (~10 GB), endpoint files and results --
budget ~30 GB.

Edit placeholders:
- `WORK=<path_to_work_folder>` in **both** `llm_server/setup_env.sh` and
  `Scripts/hpc/setup_sim_env.sh` -- must be identical (shared endpoint folder).
- `--account=<account_name>` in **both** `.slurm` files.

**Build the two venvs** from an interactive session (the login node can't be
used for this). `hpcshell` defaults to 1 CPU / 2 GB / 2 h, which is too small
for installing vLLM, so ask for more. Python comes from Spack's
`py-virtualenv` (RCD's documented method; gives Python 3.10, which vLLM and
SUMO support); all packages are plain `pip`, never `spack load`:

```bash
hpcshell --account=<account_name> --cpus-per-task=8 --mem=32G --time=03:00:00
cd <repo>/GreenEVT
mkdir -p $WORK    # if it doesn't exist yet; same path as in the setup scripts
spack load py-virtualenv

# model venv
virtualenv $WORK/llm_venv
source llm_server/setup_env.sh
pip install -r llm_server/requirements.txt
deactivate

# sim venv
virtualenv $WORK/sim_venv
source Scripts/hpc/setup_sim_env.sh
pip install -r Scripts/hpc/requirements.txt
sumo --version     # should print Eclipse SUMO 1.27.1
deactivate

# pre-download the model (also confirms the node has internet access)
source llm_server/setup_env.sh
huggingface-cli download Qwen/Qwen3-14B-AWQ
```

If `virtualenv` reports an old Python (< 3.10) or the download hangs (no
internet from compute nodes), stop and open an HPC Issues ticket / RCD
booking -- the rest of the setup depends on both.

## 2. Smoke test (does it work?)

Uses the bundled grid: 12 households commuting home -> work -> home for 2
days (48 EV legs), with a deliberately tiny battery so public-charging
decisions actually happen. Takes a few minutes once the model is up.

```bash
cd <repo>/GreenEVT/llm_server
sbatch serve_model.slurm

cd ../Scripts/hpc
sbatch --export=ALL,NET=hpc/smoke/dummy_grid.net.xml,ROUTES=hpc/smoke/routes.xml,CHARGERS=hpc/smoke/chargers.xml,DAYS=2,EXTRA_ARGS="--soc-mean 25" run_charge_eval.slurm

squeue -u $USER
```

Paths in `NET`/`ROUTES`/`CHARGERS` are relative to `GreenEVT/Scripts`
(the sim job `cd`s there). Logs: `charge-llm-vllm-<jobid>.log` in
`llm_server/`, `charge-eval-sim-<jobid>.log` in `Scripts/hpc/`.

**It works if** the sim log ends with a table and `Wrote .../summary.csv`, and:
- `ev_legs` = 48 and `completed` = 48 for all three policies
- `legs_charging_pct` > 0 for all three (charging decisions are executed)
- `llm_fallbacks` = 0, or at least small -- fallbacks mean the model's reply
  was unusable; the reason is in the `llm_fallback` column of `llm/trips.csv`
- `llm_latency_s_mean` is a number, not `nan`

Results: `$WORK/charge_eval_out/seed_42_<jobid>/` -- `summary.csv`, and per
policy `trips.csv` (one row per EV leg: start/arrival/min SoC, charger and
target, charge and queue time; for risk the household's riskiness; for llm the
prompt and the model's decision), `charging_totals.csv` (public energy per
station and home energy), and `sumo.log`.

For reference, a local run of this exact smoke test against a stub model gave
base / risk / llm `legs_charging_pct` of 12.5 / 12.5 / 20.8 and `depleted` of
2 / 2 / 0; base and risk should match that on the HPC (same seed), llm will
differ with the real model. SUMO's `Warning: Battery of vehicle ... is
depleted` lines in the log correspond to the `depleted` count.

A `Could not connect to TraCI server ... Retrying` line at the start of each
policy is normal -- SUMO is still starting up.

## 3. Real runs (Palo Alto)

Use the same ActivityGen route file (and tracker, if any) the team runs
`traci_testing.py` with, and the number of days it was generated for:

```bash
sbatch --export=ALL,NET=<PA.net.xml>,ROUTES=<activitygen .rou.xml>,TRACKER=<tracker.xml>,DAYS=5,SEED=42 run_charge_eval.slurm
```

- `CHARGERS` defaults to `./chargers_additional.xml` (in `Scripts/`).
- EVs are the vType `electric_vehicle` with the battery device params, and
  trip IDs follow ActivityGen's `<household>:<trip number>` form -- the sim
  uses them to carry battery state and detect return-home legs.
- Several seeds can run as parallel sim jobs against one model job.
- `POLICIES="base risk"` runs without the model job at all.

Calibrate after the first run:
- **Energy estimate:** decisions use `--wh-per-m 0.20` (the sim's default).
  SUMO's battery model used ~0.31-0.37 Wh/m on the smoke grid, which is why
  base runs some batteries empty there. If base shows `depleted` legs on PA,
  or `arrival_soc` well below the "predicted arrival SoC" in llm prompts,
  raise it with `EXTRA_ARGS="--wh-per-m <value>"` (applies to all three
  policies).
- **Does anyone need to charge?** Long-range batteries, short urban legs and
  overnight home charging can mean nobody ever stops at a public charger, and
  then all three policies look identical. Check `legs_charging_pct`; lower
  `--soc-mean` (first-day SoC) or use more days if it's ~0.

Other useful `EXTRA_ARGS`: `--soc-sd`, `--risk-mean`, `--risk-sd`,
`--llm-concurrency` (default 16 simultaneous requests), `--verbose` (the
sim's per-step prints; very large logs). `python charge_eval.py -h` lists
everything.

## Troubleshooting

| symptom | likely cause / fix |
|---|---|
| sim: `No model endpoint came up` | model job not running or crashed -- check `squeue` and the vLLM log; `WORK` differs between the two setup scripts |
| vLLM log: CUDA / out-of-memory at startup | lower `--gpu-memory-utilization` or `--max-model-len` in `serve_model.slurm` (the A100s are 40 GB); `squeue` should show the job on a gpunode003-012 |
| sim log: endpoint found but requests time out | the sim node can't reach the GPU node's port -- both jobs must be on `batch-warp`; if they are, ask RCD whether node-to-node traffic on high ports is allowed |
| job pending a long time | check `squeue -u $USER` REASON; `batch-warp` is low priority under fair share, so waits are normal when the cluster is busy |
| vLLM log: driver/CUDA version error | install a vLLM/torch build matching the node's driver (pin `vllm==<ver>` in `llm_server/requirements.txt`) |
| `llm_fallback` = `request_failed: ... 400` | server rejected the request; most often `tool_choice`/`enable_thinking`. For a non-Qwen3 model add `EXTRA_ARGS="--llm-thinking"`; check the vLLM log for the exact error |
| `llm_fallback` = `invalid_decision` | model picked a station not in its list or a bad target -- inspect `llm_decision` in `trips.csv` |
| sim: `ModuleNotFoundError: traci` / `sumo: not found` | sim venv not active or missing packages -- rerun step 1 for the sim venv |
| many legs `vanished` or `unfinished` | vehicles teleported out (`--time-to-teleport 300` in the sim) or the sim hit `DAYS`; check `sumo.log`, and that `DAYS` matches the ActivityGen schedule |
| LLM run slow | expected 1 call per EV leg; check `llm_latency_s_mean`; raise `--llm-concurrency` if many legs start together |

Running `traci_testing.py` directly (no `charge_policy`) behaves exactly as
before; on the smoke scenario it produces identical charging totals to
main's original version.
