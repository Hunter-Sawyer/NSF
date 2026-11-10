"""
Evaluate charge decision-making policies (base / risk / llm) in the main
multi-day ActivityGen sim (traci_testing.run_traci_simulation), then write a
side-by-side summary.

    python charge_eval.py --net <PA.net.xml> --routes <activitygen .rou.xml> \
        --days 5 --policies base risk llm --out eval_out

For llm, each model runs as its own SLURM job (see ../llm_server/). Point the
sim at them with --llm-endpoint-file <dir or .json> (the endpoint files those
jobs write) and --llm-models <hf ids>, or --llm-endpoint http://<host>:<port>/v1
for a single server. Results from several sim jobs can be merged with
combine_summaries.py.

Every policy sees the identical population: each household's first battery
level comes from an RNG seeded by (seed, household), SUMO gets the same
--seed, and riskiness is seeded per household. Battery state carries over
between a household's legs and home charging applies, exactly as in the
main sim; only the charge decision differs.
"""
import argparse
import csv
import gzip
import json
import os
import random
import statistics

import charge_policies
import traci_testing


def write_csv(path, rows, fields=None):
    """Write dict rows; ``fields`` gives the header (and order) even when empty."""
    fields = list(fields or [])
    for row in rows:
        fields += [k for k in row if k not in fields]
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2) if xs else float('nan')


def summarize(name, legs, totals, depleted_soc):
    done = [leg for leg in legs if leg["end"] in ("arrived", "stop") and "arrival_soc" in leg]
    mean = lambda xs: round(statistics.mean(xs), 2) if xs else float('nan')
    llm = [leg for leg in legs if "llm_decision" in leg or "llm_fallback" in leg]
    depleted = sum(leg["min_soc"] <= depleted_soc for leg in legs)
    n = len(legs) or 1
    return {
        'policy': name,
        'ev_legs': len(legs),
        'completed': len(done),
        'unfinished': sum(leg["end"] == "unfinished" for leg in legs),
        'vanished': sum(leg["end"] == "vanished" for leg in legs),
        'depleted': depleted,
        'depleted_pct': round(100.0 * depleted / n, 2),
        'households_depleted': len({leg["household"] for leg in legs if leg["min_soc"] <= depleted_soc}),
        'arrival_soc_mean': mean([leg["arrival_soc"] for leg in done]),
        'arrival_soc_p10': _pct([leg["arrival_soc"] for leg in done], 0.10),
        'trip_min_mean': mean([leg["trip_min"] for leg in done]),
        'legs_charging_pct': round(100.0 * sum(leg["charge_stops"] > 0 for leg in legs) / n, 2),
        'charge_min_mean': mean([leg["charge_min"] for leg in legs]),
        'wait_min_mean': mean([leg["wait_min"] for leg in legs]),
        'public_kwh': round(sum(totals["station_energy_wh"].values()) / 1000.0, 3),
        'home_kwh': round(totals["home_energy_wh"] / 1000.0, 3),
        'stations_used': sum(wh > 0 for wh in totals["station_energy_wh"].values()),
        'llm_fallbacks': sum(1 for leg in llm if leg.get("llm_fallback")),
        'llm_latency_s_mean': mean([leg["llm_latency_s"] for leg in llm
                                    if "llm_latency_s" in leg and not leg.get("llm_cached")]),
    }


def run_policy(policy, args, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    def initial_charge(household, capacity):
        rng = random.Random(f"{args.seed}:{household}")
        soc = min(100.0, max(1.0, rng.gauss(args.soc_mean, args.soc_sd)))
        return soc / 100.0 * capacity

    legs, decisions, home_sessions, station_demand = [], [], [], {}
    charge_events, station_occupancy = [], []
    try:
        totals = traci_testing.run_traci_simulation(
            tracker_file=args.tracker,
            route_file=args.routes,
            net_file=args.net,
            chargers_file=args.chargers,
            estimated_consumption_wh_per_meter=args.wh_per_m,
            simulation_days=args.days,
            charging_totals_file=os.path.join(out_dir, "charging_totals.csv"),
            charge_policy=policy,
            initial_charge_fn=initial_charge,
            trip_log=legs,
            decision_log=decisions,
            home_sessions=home_sessions,
            station_demand=station_demand,
            demand_bin_s=args.demand_bin_s,
            charge_events=charge_events,
            station_occupancy=station_occupancy,
            sumo_seed=args.seed,
            sumo_log=os.path.join(out_dir, "sumo.log"),
            verbose=args.verbose,
        )
    finally:
        if hasattr(policy, "close"):
            policy.close()

    for leg in legs:
        leg["policy"] = policy.name
        leg["min_soc"] = round(leg["min_soc"], 2)
        leg["charge_min"] = round(leg["charge_min"], 2)
        leg["wait_min"] = round(leg["wait_min"], 2)
        leg["charged_kwh"] = round(leg["charged_kwh"], 4)
        leg["depleted"] = leg["min_soc"] <= args.depleted_soc
    write_csv(os.path.join(out_dir, "trips.csv"), legs)
    # Full state + choice of every decision: replay_decisions.py feeds these
    # states to any other policy/model.
    with open(os.path.join(out_dir, "decisions.jsonl"), "w") as f:
        for i, record in enumerate(decisions):
            f.write(json.dumps({"decision_id": i, "policy": policy.name, **record},
                               sort_keys=True) + "\n")
    # Charging demand, kept at the sim's full resolution so any real-world
    # format (per charger / port / area, kW readings or kWh per interval, any
    # bin size or alignment) can be rebuilt later without re-running.
    write_csv(os.path.join(out_dir, "charge_events.csv.gz"), [
        {"time_s": t, "station": station, "vehicle_id": vehicle, "household": household,
         "energy_wh": wh} for t, station, vehicle, household, wh in charge_events],
        ["time_s", "station", "vehicle_id", "household", "energy_wh"])
    write_csv(os.path.join(out_dir, "station_occupancy.csv"), station_occupancy,
              ["time_s", "station", "vehicles", "vehicle_ids"])
    write_csv(os.path.join(out_dir, "station_demand.csv"), [
        {"bin_start_s": t, "bin_s": args.demand_bin_s, "station": station, "energy_wh": round(wh, 3),
         "avg_power_kw": round(wh / 1000.0 / (args.demand_bin_s / 3600.0), 4)}
        for (t, station), wh in sorted(station_demand.items())],
        ["bin_start_s", "bin_s", "station", "energy_wh", "avg_power_kw"])
    write_csv(os.path.join(out_dir, "stations.csv"), totals["stations"],
              ["station", "lane", "edge", "start_pos", "end_pos", "power_w", "efficiency",
               "x", "y", "lon", "lat"])
    write_csv(os.path.join(out_dir, "households.csv"), totals["households"],
              ["household", "capacity_wh", "home_edge", "x", "y", "lon", "lat"])
    write_csv(os.path.join(out_dir, "home_sessions.csv"), home_sessions,
              ["household", "home_edge", "start_s", "end_s", "energy_wh", "power_w"])
    write_csv(os.path.join(out_dir, "home_pending.csv"), totals["home_pending"],
              ["household", "arrival_s", "charge_wh", "capacity_wh", "home_edge"])
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump({"policy": policy.name, "step_length_s": totals["step_length_s"],
                   "simulation_days": args.days, "seed": args.seed,
                   "home_charge_power_w": traci_testing.HOME_CHARGE_POWER_W,
                   "home_charge_efficiency": traci_testing.HOME_CHARGE_EFFICIENCY,
                   "args": {k: v for k, v in vars(args).items() if not k.startswith("llm_endpoint")}},
                  f, indent=1, sort_keys=True)
    return summarize(policy.name, legs, totals, args.depleted_soc)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--net", required=True)
    p.add_argument("--routes", required=True, help="ActivityGen route file (trip IDs like h12c1:3)")
    p.add_argument("--chargers", default="chargers_additional.xml")
    p.add_argument("--tracker", default=None, help="optional tracker additional file")
    p.add_argument("--days", type=int, default=1,
                   help="must match the days the ActivityGen schedule was generated for")
    p.add_argument("--policies", nargs="+", default=["base", "risk", "llm"],
                   choices=["base", "risk", "llm"])
    p.add_argument("--out", default="./charge_eval_out")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--soc-mean", type=float, default=50.0,
                   help="mean first-leg SoC per household, %% (the sim's own default is 50)")
    p.add_argument("--soc-sd", type=float, default=10.0)
    p.add_argument("--wh-per-m", type=float, default=0.20,
                   help="energy estimate used for decisions (the sim's default 0.20 Wh/m)")
    p.add_argument("--depleted-soc", type=float, default=0.5,
                   help="a leg counts as depleted if SoC drops to this %% or below")
    p.add_argument("--risk-mean", type=float, default=0.5)
    p.add_argument("--risk-sd", type=float, default=0.2)
    p.add_argument("--llm-endpoint", default="http://localhost:8000/v1")
    p.add_argument("--llm-endpoint-file", default=None,
                   help="endpoint JSON written by llm_server/serve_model.slurm, or a directory "
                        "of them (newest live one is used); overrides --llm-endpoint")
    p.add_argument("--llm-concurrency", type=int, default=16,
                   help="max simultaneous LLM requests (legs starting in the same step)")
    p.add_argument("--llm-models", nargs="+", default=None,
                   help="run the llm policy once per model (Hugging Face ids, as served); each "
                        "is matched to its endpoint file. Default: the one model the server lists")
    p.add_argument("--llm-thinking", action="store_true", help="leave Qwen3 thinking mode on")
    p.add_argument("--demand-bin-s", type=int, default=60,
                   help="time bin for station_demand.csv, seconds")
    p.add_argument("--deterministic", action="store_true",
                   help="send LLM requests one at a time (no batching effects); slower")
    p.add_argument("--verbose", action="store_true", help="keep the sim's per-step prints")
    args = p.parse_args()
    if args.deterministic:
        args.llm_concurrency = 1

    # traci_testing resolves relative paths against Scripts/; make CLI paths
    # relative to where the command was run instead.
    for attr in ("net", "routes", "chargers", "tracker", "out"):
        if getattr(args, attr):
            setattr(args, attr, os.path.abspath(getattr(args, attr)))

    # (policy, model, endpoint) for each run; "llm" expands to one run per model.
    runs = []
    for name in args.policies:
        if name != "llm":
            runs.append((name, None, None))
            continue
        for model in args.llm_models or [None]:
            endpoint = args.llm_endpoint
            if args.llm_endpoint_file:
                endpoint = charge_policies.endpoint_from_file(args.llm_endpoint_file, model=model)
            print(f"Using LLM endpoint {endpoint}" + (f" for {model}" if model else ""), flush=True)
            runs.append((name, model, endpoint))

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for name, model, endpoint in runs:
        policy = charge_policies.make_policy(name, args, args.out, model=model, endpoint=endpoint)
        print(f"=== running {policy.name} ===", flush=True)
        rows.append(run_policy(policy, args, os.path.join(args.out, policy.name)))

    write_csv(os.path.join(args.out, "summary.csv"), rows)
    print_table(rows)
    print(f"\nWrote {os.path.join(args.out, 'summary.csv')}")


def print_table(rows):
    width = max(len(k) for k in rows[0])
    col = max(12, max(len(r['policy']) for r in rows) + 2)
    print("\n" + "metric".ljust(width) + "".join(r['policy'].rjust(col) for r in rows))
    for key in rows[0]:
        if key != 'policy':
            print(key.ljust(width) + "".join(str(r[key]).rjust(col) for r in rows))


if __name__ == "__main__":
    main()
