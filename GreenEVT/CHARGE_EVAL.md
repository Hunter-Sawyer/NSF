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
| `llm-<model>` | an LLM (served by vLLM) sees the leg -- time of day, SoC, whether it ends at home, and the same charger options -- and picks charge / station / target SoC. Run once per model (below) |

Models compared (all unquantized bf16, one 40 GB A100 each):

| model | Hugging Face id | results folder |
|---|---|---|
| Qwen3 4B | `Qwen/Qwen3-4B` | `llm-qwen3-4b` |
| Qwen3 8B | `Qwen/Qwen3-8B` | `llm-qwen3-8b` |
| Llama 3.2 1B | `meta-llama/Llama-3.2-1B-Instruct` | `llm-llama-3.2-1b-instruct` |
| Llama 3.2 3B | `meta-llama/Llama-3.2-3B-Instruct` | `llm-llama-3.2-3b-instruct` |
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B-Instruct` | `llm-llama-3.1-8b-instruct` |

All three get the same households, first-day battery levels (seeded per
household) and SUMO seed.

Two separate SLURM jobs:

```
llm_server/serve_model.slurm      GPU job (batch-warp, 1x 40 GB A100) -- vLLM serving one MODEL; one job per model
Scripts/hpc/run_charge_eval.slurm CPU job (batch-warp) -- SUMO + charge_eval.py, runs base, risk and each LLM_MODELS entry in turn
```

They find each other through `$WORK/vllm_endpoints/<jobid>.json`
(`{"host", "port", "model"}`, matched by model name), which the model job writes once vLLM answers
and deletes when it exits. The sim job waits (up to 60 min) for one to
appear, so both can be submitted back to back.

## Files

| file | purpose |
|---|---|
| `Scripts/traci_testing.py` | the team's multi-day sim; takes an optional `charge_policy` plus decision / leg / demand logs (default behavior unchanged) |
| `Scripts/charge_eval.py` | runs the sim once per policy, writes per-leg `trips.csv` and a side-by-side `summary.csv` |
| `Scripts/charge_policies.py` | the three policies + LLM client (stdlib HTTP only) |
| `Scripts/combine_summaries.py` | merges `summary.csv` from several sim jobs into one table |
| `Scripts/replay_decisions.py`, `Scripts/hpc/run_replay.slurm` | replays logged decision states through any policies/models (section 4) |
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
holds the venvs, the five model downloads (~65 GB in bf16), endpoint files
and results -- budget ~100 GB.

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

# Hugging Face login (needed for the gated Llama models; see below).
# Run after sourcing setup_env.sh so the token is stored under $WORK/hf_cache.
source llm_server/setup_env.sh
huggingface-cli login

# pre-download all five models (also confirms the node has internet access)
for m in Qwen/Qwen3-4B Qwen/Qwen3-8B meta-llama/Llama-3.2-1B-Instruct \
         meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.1-8B-Instruct; do
    huggingface-cli download "$m"
done
```

**Llama access (one-time, before the download).** Meta's Llama repos are
gated: on huggingface.co, sign in, open each of the three `meta-llama/...`
model pages and accept the license (approval is usually quick but not
instant), then create a read token under Settings -> Access Tokens and paste
it into `huggingface-cli login`. The Qwen models need none of this.

If `virtualenv` reports an old Python (< 3.10) or the download hangs (no
internet from compute nodes), stop and open an HPC Issues ticket / RCD
booking -- the rest of the setup depends on both.

## 2. Smoke test (does it work?)

Uses the bundled grid: 12 households commuting home -> work -> home for 2
days (48 EV legs), with a deliberately tiny battery so public-charging
decisions actually happen. Uses one model (Qwen3-4B, smallest download that
isn't gated); takes a few minutes once the model is up.

```bash
cd <repo>/GreenEVT/llm_server
sbatch --export=ALL,MODEL=Qwen/Qwen3-4B serve_model.slurm

cd ../Scripts/hpc
sbatch --export=ALL,NET=hpc/smoke/dummy_grid.net.xml,ROUTES=hpc/smoke/routes.xml,CHARGERS=hpc/smoke/chargers.xml,DAYS=2,LLM_MODELS=Qwen/Qwen3-4B,EXTRA_ARGS="--soc-mean 25" run_charge_eval.slurm

squeue -u $USER
```

Paths in `NET`/`ROUTES`/`CHARGERS` are relative to `GreenEVT/Scripts`
(the sim job `cd`s there). Logs: `charge-llm-vllm-<jobid>.log` in
`llm_server/`, `charge-eval-sim-<jobid>.log` in `Scripts/hpc/`.

**It works if** the sim log ends with a table and `Wrote .../summary.csv`, and:
- `ev_legs` = 48 and `completed` = 48 for `base`, `risk` and `llm-qwen3-4b`
- `legs_charging_pct` > 0 for all three (charging decisions are executed)
- `llm_fallbacks` = 0, or at least small -- fallbacks mean the model's reply
  was unusable; the reason is in the `llm_fallback` column of
  `llm-qwen3-4b/trips.csv`
- `llm_latency_s_mean` is a number, not `nan`

Results: `$WORK/charge_eval_out/seed_42_<jobid>/` -- `summary.csv`, and per
policy folder:

| file | contents |
|---|---|
| `trips.csv` | one row per EV leg: start/arrival/min SoC, charger and target, charge and queue time; for risk the household's riskiness; for llm the prompt and the model's decision |
| `decisions.jsonl` | the full state at every decision (SoC, capacity, time, destination, home-return, energy estimate, and every charger's routes, energy, power and occupancy) plus the choice -- input for replay (section 4) |
| `station_demand.csv` | public charging energy per station per minute (`--demand-bin-s`) |
| `home_sessions.csv` | home charging sessions: start (arrival home), end at the home charger's rate, energy |
| `home_pending.csv` | households home at the end of the sim that never left again (their overnight charge never happens in the sim) |
| `charging_totals.csv`, `sumo.log` | the sim's own totals and SUMO's log |

The last three are what a comparison with the city's real charge demand
curves will be built on.

For reference, a local run of this exact smoke test gave base / risk
`legs_charging_pct` of 14.58 / 14.58, `depleted` of 3 / 3, and `public_kwh`
of 4.927 / 3.04. Base and risk must match that exactly on the HPC (same seed,
fully deterministic); llm numbers depend on the model. SUMO's `Warning: Battery of vehicle ... is
depleted` lines in the log correspond to the `depleted` count.

A `Could not connect to TraCI server ... Retrying` line at the start of each
policy is normal -- SUMO is still starting up.

## 3. Real runs (Palo Alto)

Use the same ActivityGen route file (and tracker, if any) the team runs
`traci_testing.py` with, and the number of days it was generated for:

```bash
PA="NET=<PA.net.xml>,ROUTES=<activitygen .rou.xml>,TRACKER=<tracker.xml>,DAYS=<days>,SEED=42"
MODELS="Qwen/Qwen3-4B Qwen/Qwen3-8B meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.1-8B-Instruct"

# one GPU job per model
cd <repo>/GreenEVT/llm_server
for m in $MODELS; do sbatch --export=ALL,MODEL=$m serve_model.slurm; done

# base + risk once (no model needed), then one sim job per model, all in parallel
cd ../Scripts/hpc
sbatch --export=ALL,$PA,POLICIES="base risk" run_charge_eval.slurm
for m in $MODELS; do sbatch --export=ALL,$PA,POLICIES=llm,LLM_MODELS=$m run_charge_eval.slurm; done

# when all sim jobs are done: one table with base, risk and all five models
source setup_sim_env.sh && cd ..
python combine_summaries.py $WORK/charge_eval_out/seed_42_* --out $WORK/charge_eval_out/combined_seed_42.csv
```

Each sim job waits (up to 60 min, `LLM_WAIT_MINUTES`) for its model's server,
so submit everything at once. Every job uses the same seed, so base and risk
only need to run once, and each model sees the identical population. If GPUs
are scarce, the model jobs can also run one or two at a time -- a sim job just
waits for its model. Alternatively, one sim job can run several models in
turn: `LLM_MODELS="<id> <id>"` (all those model jobs must be up).

- `DAYS` must match the schedule: e.g. the 10k-population `6_179_rou.xml`
  covers one day (departures end around 18:15), so `DAYS=1`.
- `CHARGERS` defaults to `./chargers_additional.xml` (in `Scripts/`).
- The tracker writes to `../outputs/<name>_traffic.xml` relative to its own
  folder, so keep it in a `trackers/` folder next to an `outputs/` folder.
- EVs are the vType `electric_vehicle` with the battery device params, and
  trip IDs follow ActivityGen's `<household>:<trip number>` form -- the sim
  uses them to carry battery state and detect return-home legs.
- Several seeds can run as parallel sim jobs against the same model jobs.
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
`--llm-concurrency` (default 16 simultaneous requests), `--deterministic`,
`--verbose` (the sim's per-step prints; very large logs).
`python charge_eval.py -h` lists everything.

## 4. Comparing methods on identical states (replay)

Each closed-loop run above diverges from the others as soon as two methods
decide differently (different charging -> different SoC and traffic later).
To compare *decisions* like for like, replay one run's logged states through
every method: no SUMO, just each policy/model deciding on exactly the same
sequence of states.

```bash
cd <repo>/GreenEVT/Scripts/hpc
# e.g. every method on the states base encountered (model jobs must be up)
sbatch --export=ALL,STATES=$WORK/charge_eval_out/seed_42_<base job>/base,LLM_MODELS="$MODELS" run_replay.slurm
```

Output: `replay_<job>.csv` next to the states, one row per state with SoC,
time, home-return, direct-route arrival SoC, the logged choice, and each
method's charge / station / target side by side. The log prints each
method's charge rate and pairwise agreement on charge / don't charge, and
`matches_logged` for the run's own method, which must be 1.000 (replay
reproduces the run exactly). Replays of base and risk need no model job and
finish in seconds. Any run's states can be used (`.../risk`,
`.../llm-qwen3-8b`, ...), and `STATES` may list several.

## 5. Determinism

Everything that could vary between runs is fixed, so reruns with the same
inputs and `SEED` give identical files:
- first-day battery levels and riskiness are seeded per household; SUMO gets
  `--seed` and `--threads 1`;
- vehicles are handled in sorted ID order each step, and the sim job sets
  `PYTHONHASHSEED=0`;
- decision states are routed with a copy of the EV type that has no speed
  deviation. (SUMO's `findRoute` otherwise draws a random speed factor for a
  throwaway vehicle, which shifts every later vehicle's random draws -- a
  method that routes more would change traffic for reasons unrelated to its
  decisions.) Travel times are unaffected; the team's default path still
  routes as before.

Checked locally: rerunning under different Python hash seeds gives identical
outputs; doubling the routing queries per decision leaves the sim unchanged;
on the same state, the state-based base makes the same choice as
`choose_least_impact_charger` every time.

The LLM is the one part that can still vary: decoding is greedy
(temperature 0, fixed request seed), but GPU arithmetic can differ slightly
depending on which requests share a batch and on prefix caching. For runs
that must be reproducible, use `DETERMINISTIC=1` on the model job (no prefix
caching), `EXTRA_ARGS="--deterministic"` on the sim/replay job (one request
at a time), and give each model job only one sim job at a time. Every
response is also cached in `llm_cache.json` in the run folder, so the
decisions a run actually made are always on record and replays against that
folder's cache are exact.

## Troubleshooting

| symptom | likely cause / fix |
|---|---|
| sim: `No model endpoint came up` | model job not running or crashed -- check `squeue` and the vLLM log; `WORK` differs between the two setup scripts |
| vLLM log: CUDA / out-of-memory at startup | lower `--gpu-memory-utilization` or `--max-model-len` in `serve_model.slurm` (the A100s are 40 GB); `squeue` should show the job on a gpunode003-012 |
| sim log: endpoint found but requests time out | the sim node can't reach the GPU node's port -- both jobs must be on `batch-warp`; if they are, ask RCD whether node-to-node traffic on high ports is allowed |
| job pending a long time | check `squeue -u $USER` REASON; `batch-warp` is low priority under fair share, so waits are normal when the cluster is busy |
| vLLM log: driver/CUDA version error | install a vLLM/torch build matching the node's driver (pin `vllm==<ver>` in `llm_server/requirements.txt`) |
| vLLM log / download: `401`, `403`, `GatedRepoError` | Llama license not accepted for that repo, or no `huggingface-cli login` -- see "Llama access" in step 1 |
| `llm_fallback` = `request_failed: ... 400` | server rejected the request; check the vLLM log for the exact error. If it mentions `enable_thinking`/`chat_template_kwargs`, add `EXTRA_ARGS="--llm-thinking"` for that model's sim job |
| RCD email: low GPU utilization | expected for the 1B/3B models (they can't fill an A100); harmless unless it says the GPU was unused |
| `llm_fallback` = `invalid_decision` | model picked a station not in its list or a bad target -- inspect `llm_decision` in `trips.csv` |
| sim: `ModuleNotFoundError: traci` / `sumo: not found` | sim venv not active or missing packages -- rerun step 1 for the sim venv |
| many legs `vanished` or `unfinished` | vehicles teleported out (`--time-to-teleport 300` in the sim) or the sim hit `DAYS`; check `sumo.log`, and that `DAYS` matches the ActivityGen schedule |
| LLM run slow | expected 1 call per EV leg; check `llm_latency_s_mean`; raise `--llm-concurrency` if many legs start together |

Running `traci_testing.py` directly (no `charge_policy`) behaves exactly as
before; on the smoke scenario it produces identical charging totals to
main's original version.
