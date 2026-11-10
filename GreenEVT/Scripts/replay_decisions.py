"""
Replay logged decision states through any set of policies/models, so every
method is compared on the *identical* sequence of states (counterfactual:
"what would X have decided here?"), without re-running SUMO.

    python replay_decisions.py $WORK/charge_eval_out/seed_42_<job>/base \
        --policies base risk llm --llm-models Qwen/Qwen3-4B meta-llama/Llama-3.2-1B-Instruct \
        --llm-endpoint-file $LLM_ENDPOINT_DIR --out replay_base_states.csv

Input: run folders (or decisions.jsonl files) written by charge_eval.py.
Output: one row per logged state with each policy's decision side by side,
plus a printed summary (charge rate per policy, pairwise agreement on
charge / don't charge). Replaying a run's own policy reproduces its logged
decisions exactly; the summary reports that as ``matches_logged``.
"""
import argparse
import json
import os

import charge_policies
from charge_eval import write_csv


def load_states(paths):
    records = []
    for path in paths:
        if os.path.isdir(path):
            path = os.path.join(path, "decisions.jsonl")
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                record["source"] = path
                records.append(record)
    return records


def summarize_choice(choice, capacity):
    if choice is None:
        return {"charge": 0, "station": "", "target_soc": ""}
    return {"charge": 1, "station": choice["station"],
            "target_soc": round(choice["target_charge"] / capacity * 100.0, 2)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("states", nargs="+", help="charge_eval.py policy folders or decisions.jsonl files")
    p.add_argument("--policies", nargs="+", default=["base", "risk", "llm"],
                   choices=["base", "risk", "llm"])
    p.add_argument("--out", default="replay.csv")
    p.add_argument("--seed", type=int, default=42, help="must match the run, for risk's riskiness")
    p.add_argument("--risk-mean", type=float, default=0.5)
    p.add_argument("--risk-sd", type=float, default=0.2)
    p.add_argument("--llm-models", nargs="+", default=None)
    p.add_argument("--llm-endpoint", default="http://localhost:8000/v1")
    p.add_argument("--llm-endpoint-file", default=None)
    p.add_argument("--llm-concurrency", type=int, default=16)
    p.add_argument("--llm-thinking", action="store_true")
    p.add_argument("--deterministic", action="store_true",
                   help="send LLM requests one at a time (no batching effects); slower")
    p.add_argument("--batch", type=int, default=64, help="states decided per LLM batch")
    p.add_argument("--llm-cache-dir", default=None,
                   help="folder holding llm_cache.json; default: the run folder of the first "
                        "STATES path, so answers that run already got are reused exactly")
    args = p.parse_args()
    if args.deterministic:
        args.llm_concurrency = 1

    records = load_states(args.states)
    first = os.path.abspath(args.states[0])
    policy_dir = first if os.path.isdir(first) else os.path.dirname(first)
    cache_dir = args.llm_cache_dir or os.path.dirname(policy_dir)

    policies = []
    for name in args.policies:
        for model in (args.llm_models or [None]) if name == "llm" else [None]:
            endpoint = args.llm_endpoint
            if name == "llm" and args.llm_endpoint_file:
                endpoint = charge_policies.endpoint_from_file(args.llm_endpoint_file, model=model)
            policies.append(charge_policies.make_policy(name, args, cache_dir,
                                                        model=model, endpoint=endpoint))

    rows = []
    for r in records:
        s = r["state"]
        cap = s["capacity"]
        logged = summarize_choice(r["choice"], cap)
        b = s["baseline"]
        rows.append({
            "source": r["source"], "decision_id": r["decision_id"], "source_policy": r["policy"],
            "vehicle_id": s["vehicle_id"], "household": s["household"],
            "time_s": s["current_time"], "soc": round(s["current_charge"] / cap * 100.0, 2),
            "capacity_wh": cap, "is_home_return": s["is_home_return"],
            "direct_min": round(b["travel_time"] / 60.0, 2) if b else "",
            "direct_arrival_soc": round((s["current_charge"] - b["distance"]
                                         * s["estimated_consumption_wh_per_meter"]) / cap * 100.0, 2)
                                  if b else "",
            "logged_charge": logged["charge"], "logged_station": logged["station"],
            "logged_target_soc": logged["target_soc"],
        })

    for policy in policies:
        print(f"=== replaying {policy.name} on {len(records)} states ===", flush=True)
        resolve = getattr(policy, "resolve", None)
        for start in range(0, len(records), args.batch):
            chunk = records[start:start + args.batch]
            finishers = [policy.choose(r["state"]) for r in chunk]
            results = resolve(finishers) if resolve else [f() for f in finishers]
            for row, r, (choice, info) in zip(rows[start:], chunk, results):
                c = summarize_choice(choice, r["state"]["capacity"])
                row[f"{policy.name}_charge"] = c["charge"]
                row[f"{policy.name}_station"] = c["station"]
                row[f"{policy.name}_target_soc"] = c["target_soc"]
                if "llm_decision" in info or "llm_fallback" in info:
                    row[f"{policy.name}_fallback"] = info.get("llm_fallback", "")
        if hasattr(policy, "close"):
            policy.close()

    write_csv(args.out, rows)

    names = [p.name for p in policies]
    n = len(rows) or 1
    print(f"\n{len(rows)} states from {len(args.states)} run(s)")
    print("policy".ljust(30) + "charge_rate".rjust(12) + "matches_logged".rjust(16))
    for name in names:
        rate = sum(row[f"{name}_charge"] for row in rows) / n
        same = [row for row in rows if row["source_policy"] == name]
        match = (sum(row[f"{name}_charge"] == row["logged_charge"]
                     and row[f"{name}_station"] == row["logged_station"]
                     and row[f"{name}_target_soc"] == row["logged_target_soc"] for row in same)
                 / len(same)) if same else None
        print(name.ljust(30) + f"{rate:12.3f}" + (f"{match:16.3f}" if match is not None else "-".rjust(16)))
    if len(names) > 1:
        print("\nagreement on charge / don't charge")
        print("".ljust(30) + "".join(name[:14].rjust(16) for name in names))
        for a in names:
            print(a.ljust(30) + "".join(
                f"{sum(row[f'{a}_charge'] == row[f'{b}_charge'] for row in rows) / n:16.3f}"
                for b in names))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
