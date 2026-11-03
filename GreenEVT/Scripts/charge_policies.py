"""
Charge decision-making policies compared by charge_eval.py.

Every policy turns a trip context into a plan in routing.plan_ev_route's
format -- [(node, action, soc_after_action), ...] -- so the TraCI executor in
charge_eval.py runs all three identically and only the decision differs:

    base  -- routing.plan_ev_route + force_last_charge_to_80 (current logic)
    risk  -- same planner, but SoC buffers come from a per-driver Gaussian
             riskiness score instead of the fixed 0% / 80% rules
    llm   -- one LLM call at trip outset (vLLM on the HPC, via SSH tunnel)
             picks whether/where/how much to charge; the plan is built from
             that decision and executed as-is
"""
import hashlib
import heapq
import itertools
import json
import math
import os
import time
import urllib.error
import urllib.request

import routing


def plan_path_soc(path):
    """Lowest SoC (proxy %) a plan predicts along the way."""
    return min(step[2] for step in path)


class BasePolicy:
    name = "base"

    def assign_traits(self, rng):
        return {}

    def _plan(self, ctx, min_destination_soc=0.0):
        return routing.plan_ev_route(
            origin=ctx["origin"],
            destination=ctx["destination"],
            initial_soc=ctx["soc_pct"],
            stations_graph=ctx["graph"],
            vehicle_profile=ctx["profile"],
            real_time_status=ctx["status"],
            min_destination_soc=min_destination_soc,
            charge_nodes=ctx["charge_nodes"],
        )

    def decide(self, ctx):
        plan = routing.force_last_charge_to_80(self._plan(ctx))
        return plan, {}


class RiskPolicy(BasePolicy):
    """
    Riskiness r ~ N(mean, sd), clipped to [0, 1]; 0 = most cautious, 1 = most
    risk-tolerant. r linearly sets:
      - min_destination_soc: buffer_cautious (r=0) -> buffer_risky (r=1)
      - minimum SoC when leaving the last charger: floor_cautious -> floor_risky
        (replaces the fixed 80% rule)
    If the buffer makes the trip infeasible, it's relaxed to 0 rather than
    leaving the driver without a plan.
    """
    name = "risk"

    def __init__(self, mean=0.5, sd=0.2, buffer_cautious=30.0, buffer_risky=5.0,
                 floor_cautious=90, floor_risky=50):
        self.mean, self.sd = mean, sd
        self.buffer_cautious, self.buffer_risky = buffer_cautious, buffer_risky
        self.floor_cautious, self.floor_risky = floor_cautious, floor_risky

    def assign_traits(self, rng):
        return {"riskiness": min(1.0, max(0.0, rng.gauss(self.mean, self.sd)))}

    def decide(self, ctx):
        r = ctx["traits"]["riskiness"]
        buffer = self.buffer_cautious + r * (self.buffer_risky - self.buffer_cautious)
        floor = self.floor_cautious + r * (self.floor_risky - self.floor_cautious)
        floor = min(100, int(math.ceil(floor / 10.0)) * 10)  # planner's soc_step grid

        plan = self._plan(ctx, min_destination_soc=buffer)
        relaxed = False
        if plan is None:
            plan = self._plan(ctx)
            relaxed = plan is not None
        plan = routing.force_last_charge_to_80(plan, floor=floor)
        return plan, {"dest_buffer": round(buffer, 1), "depart_floor": floor,
                      "buffer_relaxed": relaxed}


def shortest_times(adj, source):
    """Dijkstra on driving_time. adj: {node: [(nbr, minutes, energy_wh), ...]}."""
    dist, energy, pred = {source: 0.0}, {source: 0.0}, {source: None}
    counter = itertools.count()
    pq = [(0.0, next(counter), source)]
    done = set()
    while pq:
        d, _, u = heapq.heappop(pq)
        if u in done:
            continue
        done.add(u)
        for v, minutes, wh in adj.get(u, ()):
            nd = d + minutes
            if nd < dist.get(v, math.inf):
                dist[v], energy[v], pred[v] = nd, energy[u] + wh, u
                heapq.heappush(pq, (nd, next(counter), v))
    return dist, energy, pred


def _walk(pred, end):
    nodes = []
    while end is not None:
        nodes.append(end)
        end = pred[end]
    return nodes[::-1]


SYSTEM_PROMPT = """You are an electric-vehicle driver planning one trip before you leave.
Decide whether to stop at a public charger on the way, which one, and what state of charge (SoC, %) to charge to.
You care about: (1) not running the battery empty before the destination, (2) total trip time, including detour, queueing and charging time.
Energy estimates are approximate. Respond only by calling the charge_decision tool."""

USER_TEMPLATE = """Trip context:
- Current SoC: {soc:.1f}% of a {cap_kwh:.1f} kWh battery
- Direct route (no charging): {direct_min:.1f} min, uses about {direct_pct:.1f}% SoC, predicted arrival SoC {direct_arrival:.1f}%
- Candidate chargers (sorted by detour):
{stations}

Call charge_decision with charge=false to drive directly, or charge=true with a station_id from the list and a target_soc."""

STATION_LINE = ("  * station_id={sid}: detour +{detour:.1f} min, SoC on arrival {arr:.1f}%, "
                "needs {after:.1f}% SoC from there to destination, {kw:.0f} kW "
                "(~{rate:.1f} min per 10% SoC), {occ}/{stalls} stalls occupied, "
                "est. queue wait {wait:.0f} min")

CHARGE_TOOL = {
    "type": "function",
    "function": {
        "name": "charge_decision",
        "description": "Your charging decision for this trip.",
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


class LLMPolicy(BasePolicy):
    """
    One OpenAI-compatible chat call per trip (vLLM via `ssh -L 8000:gpunodeXXX:PORT`).
    The decision is executed even if the energy proxy predicts it runs out --
    that's the behavior being evaluated. Only malformed/failed responses fall
    back to BasePolicy, and that's recorded as llm_fallback.
    """
    name = "llm"

    def __init__(self, endpoint="http://localhost:8000/v1", model=None, max_candidates=5,
                 timeout=120, cache_path=None, disable_thinking=True):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.model = model or self._detect_model()
        self.max_candidates = max_candidates
        self.disable_thinking = disable_thinking
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                self.cache = json.load(f)

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
                f"Can't reach the model at {self.endpoint} ({e}). Is the SLURM job running "
                f"and the SSH tunnel up (ssh -L 8000:gpunodeXXX:PORT ...)?") from e

    def save_cache(self):
        if self.cache_path:
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f, indent=1)

    def _candidates(self, ctx):
        world = ctx["world"]
        cap = ctx["capacity_wh"]
        origin, dest, soc = ctx["origin"], ctx["destination"], ctx["soc_pct"]
        fwd = shortest_times(world.adj, origin)
        bwd = shortest_times(world.radj, dest)
        if dest not in fwd[0]:
            return None, []
        direct = {"minutes": fwd[0][dest], "pct": fwd[1][dest] / cap * 100,
                  "nodes": _walk(fwd[2], dest)}

        cands = []
        for node, sid in world.node_station.items():
            if node not in fwd[0] or node not in bwd[0]:
                continue
            to_min, to_pct = fwd[0][node], fwd[1][node] / cap * 100
            from_min, from_pct = bwd[0][node], bwd[1][node] / cap * 100
            status = ctx["status"].get(node, {})
            power_kw = world.station_power_w[sid] / 1000.0
            cands.append({
                "node": node, "sid": sid,
                "detour": to_min + from_min - direct["minutes"],
                "arr": soc - to_pct, "after": from_pct, "kw": power_kw,
                "rate": ctx["profile"]["charge_curve_at"](node, 0, 10),
                "occ": status.get("occupied", 0), "stalls": status.get("stalls", 1),
                "wait": routing._estimate_wait(status or None),
                "nodes_to": _walk(fwd[2], node),
                "nodes_from": _walk(bwd[2], node)[::-1],  # reverse-graph walk runs dest -> node
            })
        cands.sort(key=lambda c: c["detour"])
        return direct, cands[:self.max_candidates]

    def _messages(self, ctx, direct, cands):
        stations = "\n".join(STATION_LINE.format(**c) for c in cands) or "  (none reachable)"
        user = USER_TEMPLATE.format(
            soc=ctx["soc_pct"], cap_kwh=ctx["capacity_wh"] / 1000.0,
            direct_min=direct["minutes"], direct_pct=direct["pct"],
            direct_arrival=ctx["soc_pct"] - direct["pct"], stations=stations)
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]

    def _ask(self, messages):
        key = hashlib.sha256(json.dumps([self.model, messages]).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key], 0.0, True
        payload = {
            "model": self.model, "messages": messages, "temperature": 0,
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

    def _path_from_nodes(self, ctx, nodes, soc, path):
        graph = ctx["graph"]
        for prev, nxt in zip(nodes, nodes[1:]):
            soc -= graph[prev][nxt]["soc_cost"]
            path.append((nxt, "drive_to", round(soc, 1)))
        return soc

    def decide(self, ctx):
        direct, cands = self._candidates(ctx)
        if direct is None:
            return None, {"llm_fallback": "no_route"}
        messages = self._messages(ctx, direct, cands)
        info = {"llm_prompt": messages[1]["content"]}
        try:
            decision, latency, cached = self._ask(messages)
        except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError) as e:
            plan, _ = super().decide(ctx)
            return plan, {**info, "llm_fallback": f"request_failed: {e}"}
        info.update(llm_latency_s=round(latency, 3), llm_cached=cached,
                    llm_decision=json.dumps(decision))

        by_sid = {c["sid"]: c for c in cands}
        charge = bool(decision.get("charge"))
        sid = decision.get("station_id")
        target = decision.get("target_soc")
        if charge and (sid not in by_sid or not isinstance(target, int) or not 0 <= target <= 100):
            plan, _ = super().decide(ctx)
            return plan, {**info, "llm_fallback": f"invalid_decision: {decision}"}

        soc = ctx["soc_pct"]
        path = [(ctx["origin"], "start", soc)]
        if not charge:
            self._path_from_nodes(ctx, direct["nodes"], soc, path)
        else:
            c = by_sid[sid]
            soc = self._path_from_nodes(ctx, c["nodes_to"], soc, path)
            target = max(target, math.ceil(soc))  # can't "charge" downward
            path.append((c["node"], f"charge_to_{target}%_immediately", target))
            self._path_from_nodes(ctx, c["nodes_from"], target, path)
        return {"path": path, "total_time_minutes": None}, info


def endpoint_from_file(path, timeout=10):
    """
    Base URL from an endpoint JSON written by llm_server/serve_model.slurm
    ({"host", "port", ...}). Given a directory, tries its *.json files newest
    first and returns the first server that answers /v1/models -- files from
    jobs killed without running their EXIT trap are skipped that way.
    """
    if os.path.isdir(path):
        files = sorted((os.path.join(path, f) for f in os.listdir(path) if f.endswith(".json")),
                       key=os.path.getmtime, reverse=True)
    else:
        files = [path]
    for f in files:
        with open(f) as fh:
            info = json.load(fh)
        url = f"http://{info['host']}:{info['port']}/v1"
        try:
            urllib.request.urlopen(url + "/models", timeout=timeout).close()
            return url
        except (urllib.error.URLError, OSError):
            continue
    raise RuntimeError(f"No reachable vLLM endpoint in {path} -- is the model job running (squeue -u $USER)?")


def make_policy(name, args, out_dir):
    if name == "base":
        return BasePolicy()
    if name == "risk":
        return RiskPolicy(mean=args.risk_mean, sd=args.risk_sd)
    if name == "llm":
        return LLMPolicy(endpoint=args.llm_endpoint, model=args.llm_model,
                         cache_path=os.path.join(out_dir, "llm_cache.json"),
                         disable_thinking=not args.llm_thinking)
    raise ValueError(f"unknown policy {name!r}")
