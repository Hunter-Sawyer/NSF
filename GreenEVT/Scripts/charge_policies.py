"""
Charge decision-making policies for traci_testing.run_traci_simulation
(the multi-day ActivityGen sim), compared by charge_eval.py.

Each policy decides, at the start of every EV leg, whether to stop at a
public charger, which one, and the target charge -- the same decision
traci_testing.choose_least_impact_charger makes by default:

    base  -- choose_least_impact_charger as-is (10% reserve, charge to 90%)
    risk  -- the same rule, but each household's Gaussian riskiness score
             sets its reserve and charge target
    llm   -- an LLM (vLLM on the HPC) sees the leg and the same charger
             candidates and makes the call

Every policy decides only from a decision state (traci_testing.decision_state:
SoC, time, destination, and each charger's routes, energy, power and
occupancy), never from the live sim. The same code therefore runs during a
simulation and in replay_decisions.py on states logged by any earlier run,
so all methods can be compared on the identical sequence of states.

Interface: ``choose(state)`` returns a zero-argument ``finish`` callable
giving ``(choice, info)``; ``choice`` is None (drive on) or a dict with
``station``, ``edge``, ``target_charge`` (Wh), ``added_time``,
``added_distance``, and ``info`` is logged with the decision.
``resolve(finishers)`` runs the finishers -- concurrently for the LLM.
"""
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def least_impact_from_state(state, reserve_soc, charge_target_soc):
    """traci_testing.choose_least_impact_charger, computed from a decision state.

    Same arithmetic in the same order as the team's function, so given the
    same measurements it returns the same charger and target.
    """
    baseline = state["baseline"]
    if baseline is None:
        return None
    charge, capacity = state["current_charge"], state["capacity"]
    wh_per_m = state["estimated_consumption_wh_per_meter"]
    reserve_energy = capacity * reserve_soc
    required_energy = baseline["distance"] * wh_per_m + reserve_energy
    if charge >= required_energy:
        return None

    best = None
    for station in state["stations"]:
        if charge < station["distance_to_station"] * wh_per_m + reserve_energy:
            continue
        candidate_distance = station["distance_to_station"] + station["from_length"]
        candidate_time = station["to_travel_time"] + station["from_travel_time"]
        added_distance = max(0.0, candidate_distance - baseline["distance"])
        added_time = max(0.0, candidate_time - baseline["travel_time"])
        target_charge = max(capacity * charge_target_soc,
                            station["from_length"] * wh_per_m + reserve_energy)
        target_charge = min(capacity, target_charge)
        score = (added_time, added_distance, candidate_time, station["station"])
        if best is None or score < best["score"]:
            best = {"station": station["station"], "edge": station["edge"],
                    "added_time": added_time, "added_distance": added_distance,
                    "target_charge": target_charge, "score": score}
    return best


class BasePolicy:
    name = "base"

    def choose(self, state):
        choice = least_impact_from_state(state, state["reserve_soc"], state["charge_target_soc"])
        return lambda: (choice, {})


class RiskPolicy:
    """
    Riskiness r ~ N(mean, sd), clipped to [0, 1], drawn once per household
    (seeded, so every run sees the same drivers); 0 = most cautious. r sets:
      - reserve kept on arrival: reserve_cautious (r=0) -> reserve_risky (r=1)
      - charge target at the charger: target_cautious -> target_risky
    """
    name = "risk"

    def __init__(self, seed=42, mean=0.5, sd=0.2, reserve_cautious=0.30, reserve_risky=0.05,
                 target_cautious=0.90, target_risky=0.50):
        self.seed, self.mean, self.sd = seed, mean, sd
        self.reserve_cautious, self.reserve_risky = reserve_cautious, reserve_risky
        self.target_cautious, self.target_risky = target_cautious, target_risky
        self.riskiness = {}

    def _riskiness(self, household):
        if household not in self.riskiness:
            rng = random.Random(f"{self.seed}:{household}:risk")
            self.riskiness[household] = min(1.0, max(0.0, rng.gauss(self.mean, self.sd)))
        return self.riskiness[household]

    def choose(self, state):
        r = self._riskiness(state["household"])
        reserve = self.reserve_cautious + r * (self.reserve_risky - self.reserve_cautious)
        target = self.target_cautious + r * (self.target_risky - self.target_cautious)
        choice = least_impact_from_state(state, reserve, target)
        info = {"riskiness": round(r, 3), "reserve_soc": round(reserve, 3),
                "charge_target_soc": round(target, 3)}
        return lambda: (choice, info)


SYSTEM_PROMPT = """You are an electric-vehicle driver about to start one leg of your day.
Decide whether to stop at a public charger on this leg, which one, and what state of charge (SoC, %) to charge to.
You care about: (1) not running the battery empty, on this leg or the ones after it today, (2) total time, including detour, queueing and charging time.
Energy estimates are approximate. Respond only by calling the charge_decision tool."""

USER_TEMPLATE = """Leg context:
- Time of day: {clock}
- Current SoC: {soc:.1f}% of a {cap_kwh:.1f} kWh battery
- This leg ends {destination}
- Driving straight there: {direct_min:.1f} min, uses about {direct_pct:.1f}% SoC, predicted arrival SoC {direct_arrival:.1f}%
- Public chargers you could stop at (sorted by detour):
{stations}

Call charge_decision with charge=false to drive straight there, or charge=true with a station_id from the list and a target_soc."""

STATION_LINE = ("  * station_id={sid}: detour +{detour:.1f} min, SoC on arrival {arr:.1f}%, "
                "needs {after:.1f}% SoC from there to your destination, {kw:.0f} kW "
                "(~{rate:.1f} min per 10% SoC), {occ} vehicle(s) there now")

CHARGE_TOOL = {
    "type": "function",
    "function": {
        "name": "charge_decision",
        "description": "Your charging decision for this leg.",
        "parameters": {
            "type": "object",
            "properties": {
                "charge": {"type": "boolean"},
                "station_id": {"type": ["string", "null"]},
                "target_soc": {"type": "integer", "minimum": 0, "maximum": 100},
                "rationale": {"type": "string"},
            },
            "required": ["charge", "station_id", "target_soc", "rationale"],
        },
    },
}


class LLMPolicy:
    """
    One OpenAI-compatible chat call per EV leg, sent concurrently for legs
    starting in the same step. The decision is executed even if the energy
    estimate says it runs out -- that's the behavior being evaluated. Only
    failed or malformed responses fall back to the base rule (recorded as
    llm_fallback).
    """
    name = "llm"

    def __init__(self, endpoint="http://localhost:8000/v1", model=None, max_candidates=5,
                 timeout=120, cache_path=None, disable_thinking=True, concurrency=16):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.model = model or self._detect_model()
        # e.g. "llm-qwen3-8b": keeps each model's results apart in one eval.
        self.name = "llm-" + self.model.rsplit("/", 1)[-1].lower()
        self.max_candidates = max_candidates
        self.disable_thinking = disable_thinking
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                self.cache = json.load(f)
        self.pool = ThreadPoolExecutor(concurrency)

    def _http(self, path, payload=None):
        req = urllib.request.Request(
            self.endpoint + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _detect_model(self):
        try:
            return self._http("/models")["data"][0]["id"]
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(
                f"Can't reach the model at {self.endpoint} ({e}). Is the model job "
                f"running (squeue -u $USER)?") from e

    def close(self):
        self.pool.shutdown()
        if self.cache_path:
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f, indent=1)

    def resolve(self, finishers):
        if len(finishers) <= 1:
            return [finish() for finish in finishers]
        return list(self.pool.map(lambda finish: finish(), finishers))

    def choose(self, state):
        cap, charge = state["capacity"], state["current_charge"]
        wh_per_m = state["estimated_consumption_wh_per_meter"]
        baseline = state["baseline"]
        if baseline is None:
            return lambda: (None, {"llm_fallback": "no_route"})
        fallback = least_impact_from_state(state, state["reserve_soc"], state["charge_target_soc"])

        # Chargers the car can reach at all (no reserve), nearest detour first.
        cands = []
        for st in state["stations"]:
            if charge < st["distance_to_station"] * wh_per_m:
                continue
            cands.append(dict(st, added_time=max(
                0.0, st["to_travel_time"] + st["from_travel_time"] - baseline["travel_time"]),
                added_distance=max(
                0.0, st["distance_to_station"] + st["from_length"] - baseline["distance"])))
        cands = sorted(cands, key=lambda c: c["added_time"])[:self.max_candidates]

        pct = lambda wh: wh / cap * 100.0
        lines = []
        for c in cands:
            power = c["power_w"] or 22000.0
            lines.append(STATION_LINE.format(
                sid=c["station"], detour=c["added_time"] / 60.0,
                arr=pct(charge - c["distance_to_station"] * wh_per_m),
                after=pct(c["from_length"] * wh_per_m),
                kw=power / 1000.0, rate=cap * 0.1 / (power * 0.95) * 60.0,
                occ=c["vehicles_at_station"]))
        t = int(state["current_time"])
        baseline_energy = baseline["distance"] * wh_per_m
        user = USER_TEMPLATE.format(
            clock=f"day {t // 86400 + 1}, {t % 86400 // 3600:02d}:{t % 3600 // 60:02d}",
            soc=pct(charge), cap_kwh=cap / 1000.0,
            destination=(f"at home, where you can charge overnight at "
                         f"{state['home_charge_power_w'] / 1000:.1f} kW" if state["is_home_return"]
                         else "away from home (no home charger there)"),
            direct_min=baseline["travel_time"] / 60.0, direct_pct=pct(baseline_energy),
            direct_arrival=pct(charge - baseline_energy),
            stations="\n".join(lines) or "  (none reachable)")
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user}]
        by_sid = {c["station"]: c for c in cands}

        def finish():
            info = {"llm_prompt": user}
            try:
                decision, latency, cached = self._ask(messages)
            except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError) as e:
                return fallback, {**info, "llm_fallback": f"request_failed: {e}"}
            info.update(llm_latency_s=round(latency, 3), llm_cached=cached,
                        llm_decision=json.dumps(decision))
            if not decision.get("charge"):
                return None, info
            sid, target = decision.get("station_id"), decision.get("target_soc")
            if sid not in by_sid or not isinstance(target, int) or not 0 <= target <= 100:
                return fallback, {**info, "llm_fallback": f"invalid_decision: {decision}"}
            c = by_sid[sid]
            return {"station": sid, "edge": c["edge"], "target_charge": target / 100.0 * cap,
                    "added_time": c["added_time"], "added_distance": c["added_distance"]}, info

        return finish

    def _ask(self, messages):
        key = hashlib.sha256(json.dumps([self.model, messages]).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key], 0.0, True
        payload = {
            # Greedy decoding plus a fixed seed; see CHARGE_EVAL.md on determinism.
            "model": self.model, "messages": messages, "temperature": 0, "seed": 0,
            "max_tokens": 512, "tools": [CHARGE_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "charge_decision"}},
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        t0 = time.time()
        resp = self._http("/chat/completions", payload)
        latency = time.time() - t0
        msg = resp["choices"][0]["message"]
        if msg.get("tool_calls"):
            decision = json.loads(msg["tool_calls"][0]["function"]["arguments"])
        else:
            decision = json.loads(msg.get("content") or "")
        self.cache[key] = decision
        return decision, latency, False


def endpoint_from_file(path, model=None, timeout=10):
    """
    Base URL from an endpoint JSON written by llm_server/serve_model.slurm
    ({"host", "port", "model", ...}). Given a directory, tries its *.json
    files newest first and returns the first server that answers /v1/models
    -- files from jobs killed without running their EXIT trap are skipped
    that way. With ``model``, only that model's servers are considered.
    """
    if os.path.isdir(path):
        files = sorted((os.path.join(path, f) for f in os.listdir(path) if f.endswith(".json")),
                       key=os.path.getmtime, reverse=True)
    else:
        files = [path]
    for f in files:
        with open(f) as fh:
            info = json.load(fh)
        if model is not None and info.get("model") != model:
            continue
        url = f"http://{info['host']}:{info['port']}/v1"
        try:
            urllib.request.urlopen(url + "/models", timeout=timeout).close()
            return url
        except (urllib.error.URLError, OSError):
            continue
    which = f"{model} endpoint" if model else "vLLM endpoint"
    raise RuntimeError(f"No reachable {which} in {path} -- is the model job running (squeue -u $USER)?")


def make_policy(name, args, out_dir, model=None, endpoint=None):
    if name == "base":
        return BasePolicy()
    if name == "risk":
        return RiskPolicy(seed=args.seed, mean=args.risk_mean, sd=args.risk_sd)
    if name == "llm":
        return LLMPolicy(endpoint=endpoint or args.llm_endpoint, model=model,
                         cache_path=os.path.join(out_dir, "llm_cache.json"),
                         disable_thinking=not args.llm_thinking,
                         concurrency=args.llm_concurrency)
    raise ValueError(f"unknown policy {name!r}")
