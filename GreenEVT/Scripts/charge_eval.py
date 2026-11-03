"""
Evaluate charge decision-making policies (base / risk / llm) on the same
SUMO scenario and population, then write a side-by-side summary.

    python charge_eval.py --net PA.network.net.xml --routes test_PA_rou.xml \
        --chargers chargers_additional.xml --policies base risk llm --out eval_out

For llm, the model runs as its own SLURM job (see ../llm_server/). Point the
sim at it with --llm-endpoint-file <dir or .json> (the endpoint file that job
writes), or --llm-endpoint http://<host>:<port>/v1 directly.

Each vehicle's initial SoC and riskiness come from an RNG seeded by
(seed, veh_id), so every policy sees the identical population. All policies
are executed by the same code path: the plan is turned into a full SUMO
route with charging-station stops at departure, and a stopped vehicle is
released once it reaches its target SoC.
"""
import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

import sumolib
import traci

import charge_policies
import routing

CS_PREFIX = "CS:"


class World:
    """Junction graph with each charging station spliced in as its own node.

    Station `sid` on edge E (A -> B) becomes node CS:sid with A -> CS:sid
    costing E, and CS:sid -> B free (the car is already on E). Only CS nodes
    are offered to the planner as charge locations.
    """

    def __init__(self, net, wh_per_km, stations):
        self.net = net
        self.base = {}  # node -> {nbr: {'driving_time', 'energy_wh', 'sumo_edge_id'}}
        for node in net.getNodes():
            self.base[node.getID()] = {}
        for edge in net.getEdges():
            if not edge.allows("passenger"):
                continue
            a, b = edge.getFromNode().getID(), edge.getToNode().getID()
            self.base[a][b] = {
                'driving_time': edge.getLength() / edge.getSpeed() / 60.0,
                'energy_wh': edge.getLength() / 1000.0 * wh_per_km,
                'sumo_edge_id': edge.getID(),
            }

        self.node_station, self.station_power_w, self.station_edge = {}, {}, {}
        for sid, info in stations.items():
            edge = net.getEdge(info['edge'])
            node = CS_PREFIX + sid
            a, b = edge.getFromNode().getID(), edge.getToNode().getID()
            self.base[a][node] = {
                'driving_time': edge.getLength() / edge.getSpeed() / 60.0,
                'energy_wh': edge.getLength() / 1000.0 * wh_per_km,
                'sumo_edge_id': edge.getID(),
            }
            self.base[node] = {b: {'driving_time': 0.0, 'energy_wh': 0.0, 'sumo_edge_id': None}}
            self.node_station[node] = sid
            self.station_power_w[sid] = info['power_w']
            self.station_edge[sid] = edge.getID()
        self.charge_nodes = set(self.node_station)

        self.adj = {u: [(v, d['driving_time'], d['energy_wh']) for v, d in nbrs.items()]
                    for u, nbrs in self.base.items()}
        self.radj = {u: [] for u in self.base}
        for u, nbrs in self.base.items():
            for v, d in nbrs.items():
                self.radj.setdefault(v, []).append((u, d['driving_time'], d['energy_wh']))
        self._graphs = {}

    def graph_for(self, capacity_wh):
        """plan_ev_route-format graph (soc_cost in % of this capacity), cached per capacity."""
        if capacity_wh not in self._graphs:
            self._graphs[capacity_wh] = {
                u: {v: {'driving_time': d['driving_time'],
                        'soc_cost': d['energy_wh'] / capacity_wh * 100.0,
                        'sumo_edge_id': d['sumo_edge_id']}
                    for v, d in nbrs.items()}
                for u, nbrs in self.base.items()}
        return self._graphs[capacity_wh]

    def profile_for(self, capacity_wh, efficiency=0.95):
        """Linear charge curve matching SUMO's constant-power station charging."""
        def charge_curve_at(node, from_soc, to_soc):
            power = self.station_power_w[self.node_station[node]] * efficiency
            return max(0.0, to_soc - from_soc) / 100.0 * capacity_wh / power * 60.0
        return {'charge_curve_func': routing.standard_ev_charge_curve,
                'charge_curve_at': charge_curve_at}


def read_stations():
    stations = {}
    for sid in traci.chargingstation.getIDList():
        lane = traci.chargingstation.getLaneID(sid)
        try:
            power = traci.chargingstation.getChargingPower(sid)
        except (AttributeError, traci.TraCIException):
            power = float(traci.simulation.getParameter(sid, "chargingStation.power") or 22000)
        start = traci.chargingstation.getStartPos(sid)
        end = traci.chargingstation.getEndPos(sid)
        stations[sid] = {
            'edge': traci.lane.getEdgeID(lane), 'power_w': power,
            'stalls': max(1, int((end - start) // 7.5)),  # ~one car length per stall
        }
    return stations


def plan_to_sumo(plan_path, graph, world, first_edge):
    """Plan steps -> (full SUMO edge list, [(station_id, target_soc), ...])."""
    edges, stops = [first_edge], []
    prev = plan_path[0][0]
    for node, action, soc in plan_path[1:]:
        if action == 'drive_to':
            edge = graph[prev][node]['sumo_edge_id']
            if edge is not None:
                edges.append(edge)
            prev = node
        elif action.startswith('charge_to_'):
            stops.append((world.node_station[node], soc))
    return edges, stops


def soc_pct(veh_id, capacity):
    return float(traci.vehicle.getParameter(veh_id, "device.battery.chargeLevel")) / capacity * 100.0


def run_policy(policy, args, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [args.sumo_binary, "--net-file", args.net, "--route-files", args.routes,
           "--additional-files", args.chargers,
           "--chargingstations-output", os.path.join(out_dir, "charging_stations_output.xml"),
           "--time-to-teleport", "-1", "--seed", str(args.seed), "--no-step-log", "true",
           "--no-warnings", "true"]
    traci.start(cmd, label=policy.name)

    stations = read_stations()
    net = sumolib.net.readNet(args.net)
    world = World(net, args.wh_per_km, stations)
    session_minutes = {sid: [] for sid in stations}
    peak_occupancy = {sid: 0 for sid in stations}

    trips, active = {}, {}
    pool = ThreadPoolExecutor(args.llm_concurrency) if policy.name == "llm" else None
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0 and step < args.max_steps:
        traci.simulationStep()
        step += 1
        now = traci.simulation.getTime()

        status = {}
        for sid, info in stations.items():
            occupied = traci.chargingstation.getVehicleCount(sid)
            peak_occupancy[sid] = max(peak_occupancy[sid], occupied)
            done = session_minutes[sid]
            status[CS_PREFIX + sid] = {'stalls': info['stalls'], 'occupied': occupied,
                                       'avg_session_minutes': statistics.mean(done) if done else 30.0}

        # Departures are handled in three passes so every decision this step can
        # run concurrently (one LLM request each) before any TraCI calls are made.
        pending = []
        for veh_id in traci.simulation.getDepartedIDList():
            if traci.vehicle.getTypeID(veh_id) != args.ev_type:
                continue
            capacity = float(traci.vehicle.getParameter(veh_id, "device.battery.capacity"))
            rng = random.Random(f"{args.seed}:{veh_id}")
            start_soc = min(100.0, max(1.0, rng.gauss(args.soc_mean, args.soc_sd)))
            traits = policy.assign_traits(random.Random(f"{args.seed}:{veh_id}:traits"))
            traci.vehicle.setParameter(veh_id, "device.battery.chargeLevel",
                                       str(start_soc / 100.0 * capacity))

            route = traci.vehicle.getRoute(veh_id)
            trip = {'veh_id': veh_id, 'policy': policy.name, 'depart_s': now,
                    'capacity_kwh': capacity / 1000.0, 'start_soc': round(start_soc, 2),
                    'charge_stops': 0, 'charge_min': 0.0, 'queue_min': 0.0,
                    'energy_charged_kwh': 0.0, 'min_soc': start_soc, 'plan_status': 'ok',
                    **traits}
            trips[veh_id] = trip

            origin = net.getEdge(route[0]).getToNode().getID()
            destination = net.getEdge(route[-1]).getToNode().getID()
            ctx = None
            if len(route) > 1 and origin != destination:
                ctx = {'origin': origin, 'destination': destination, 'soc_pct': start_soc,
                       'capacity_wh': capacity, 'graph': world.graph_for(capacity),
                       'profile': world.profile_for(capacity), 'status': status,
                       'charge_nodes': world.charge_nodes, 'traits': traits, 'world': world}
            pending.append((veh_id, route, capacity, ctx))

        def timed_decide(ctx):
            t0 = time.time()
            plan, info = policy.decide(ctx)
            return plan, info, round(time.time() - t0, 3)

        to_decide = [ctx for *_, ctx in pending if ctx is not None]
        decisions = iter(list(pool.map(timed_decide, to_decide)) if pool and len(to_decide) > 1
                         else [timed_decide(ctx) for ctx in to_decide])

        for veh_id, route, capacity, ctx in pending:
            trip = trips[veh_id]
            stops = []
            if ctx is not None:
                plan, info, trip['decide_s'] = next(decisions)
                trip.update(info)
                if plan is None:
                    trip['plan_status'] = 'infeasible'  # keep SUMO's own route
                else:
                    trip['planned_min_soc'] = round(charge_policies.plan_path_soc(plan['path']), 2)
                    edges, stops = plan_to_sumo(plan['path'], ctx['graph'], world, route[0])
                    try:
                        traci.vehicle.setRoute(veh_id, edges)
                        for sid, _ in stops:
                            traci.vehicle.setChargingStationStop(veh_id, sid, duration=1e6)
                    except traci.TraCIException as e:
                        trip['plan_status'] = f'exec_error: {e}'
                        stops = []
            else:
                trip['plan_status'] = 'trivial'
            trip['planned_stops'] = json.dumps(stops)
            active[veh_id] = {'capacity': capacity, 'stops': stops, 'stop_start': None,
                              'soc_at_stop': None, 'last_soc': trip['start_soc'],
                              'last_energy': 0.0, 'last_dist': 0.0}

        for veh_id in traci.simulation.getArrivedIDList():
            if veh_id in active:
                st = active.pop(veh_id)
                trip = trips[veh_id]
                trip.update(arrive_s=now, trip_min=round((now - trip['depart_s']) / 60.0, 2),
                            arrival_soc=round(st['last_soc'], 2), distance_km=st['last_dist'] / 1000.0,
                            energy_used_kwh=st['last_energy'] / 1000.0)

        for veh_id, st in active.items():
            trip = trips[veh_id]
            if step % args.sample_every == 0 or st['stops']:
                st['last_soc'] = soc_pct(veh_id, st['capacity'])
                st['last_dist'] = traci.vehicle.getDistance(veh_id)
                st['last_energy'] = float(traci.vehicle.getParameter(
                    veh_id, "device.battery.totalEnergyConsumed"))
                trip['min_soc'] = min(trip['min_soc'], st['last_soc'])
            if not st['stops']:
                continue

            sid, target = st['stops'][0]
            stopped_here = (traci.vehicle.getStopState(veh_id) & 1 and
                            traci.vehicle.getRoadID(veh_id) == world.station_edge[sid])
            if st['stop_start'] is None:
                if stopped_here:
                    st['stop_start'], st['soc_at_stop'] = now, st['last_soc']
                elif (traci.vehicle.getRoadID(veh_id) == world.station_edge[sid]
                      and traci.vehicle.getSpeed(veh_id) < 0.1):
                    trip['queue_min'] += traci.simulation.getDeltaT() / 60.0
            elif st['last_soc'] >= target:
                minutes = (now - st['stop_start']) / 60.0
                trip['charge_min'] += minutes
                trip['charge_stops'] += 1
                trip['energy_charged_kwh'] += (st['last_soc'] - st['soc_at_stop']) / 100.0 * st['capacity'] / 1000.0
                session_minutes[sid].append(minutes)
                st['stops'].pop(0)
                st['stop_start'] = None
                traci.vehicle.resume(veh_id)

    for veh_id in active:
        trips[veh_id]['plan_status'] += ';unfinished'
    traci.close()
    if pool:
        pool.shutdown()
    if hasattr(policy, 'save_cache'):
        policy.save_cache()

    for trip in trips.values():
        trip['depleted'] = trip['min_soc'] <= args.depleted_soc
        trip['min_soc'] = round(trip['min_soc'], 2)
        trip['charge_min'] = round(trip['charge_min'], 2)
        trip['queue_min'] = round(trip['queue_min'], 2)

    write_csv(os.path.join(out_dir, "trips.csv"), list(trips.values()))
    write_csv(os.path.join(out_dir, "stations.csv"), [
        {'station': sid, 'power_kw': s['power_w'] / 1000.0, 'stalls': s['stalls'],
         'sessions': len(session_minutes[sid]), 'peak_occupancy': peak_occupancy[sid],
         'mean_session_min': round(statistics.mean(session_minutes[sid]), 2) if session_minutes[sid] else ''}
        for sid, s in stations.items()])
    return summarize(policy.name, list(trips.values()), peak_occupancy)


def write_csv(path, rows):
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float('nan')


def summarize(name, trips, peak_occupancy):
    done = [t for t in trips if 'arrival_soc' in t]
    mean = lambda xs: round(statistics.mean(xs), 2) if xs else float('nan')
    llm = [t for t in trips if 'llm_decision' in t or 'llm_fallback' in t]
    km = sum(t['distance_km'] for t in done)
    return {
        'policy': name,
        'ev_trips': len(trips),
        'completed': len(done),
        'unfinished': len(trips) - len(done),
        'depleted': sum(t['depleted'] for t in trips),
        'depleted_pct': round(100.0 * sum(t['depleted'] for t in trips) / len(trips), 2) if trips else 0,
        'infeasible_plans': sum(t['plan_status'].startswith('infeasible') for t in trips),
        'arrival_soc_mean': mean([t['arrival_soc'] for t in done]),
        'arrival_soc_p10': round(_pct([t['arrival_soc'] for t in done], 0.10), 2),
        'trip_min_mean': mean([t['trip_min'] for t in done]),
        'trips_charging_pct': round(100.0 * sum(t['charge_stops'] > 0 for t in trips) / len(trips), 2) if trips else 0,
        'charge_min_mean': mean([t['charge_min'] for t in trips]),
        'queue_min_mean': mean([t['queue_min'] for t in trips]),
        'energy_charged_kwh': round(sum(t['energy_charged_kwh'] for t in trips), 2),
        'peak_station_occupancy': max(peak_occupancy.values(), default=0),
        'sim_wh_per_km': round(1000 * sum(t['energy_used_kwh'] for t in done) / km, 1) if km else '',
        'llm_fallbacks': sum(1 for t in llm if t.get('llm_fallback')),
        'llm_latency_s_mean': mean([t['llm_latency_s'] for t in llm
                                    if 'llm_latency_s' in t and not t.get('llm_cached')]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--net", required=True)
    p.add_argument("--routes", required=True)
    p.add_argument("--chargers", default="./chargers_additional.xml")
    p.add_argument("--policies", nargs="+", default=["base", "risk", "llm"],
                   choices=["base", "risk", "llm"])
    p.add_argument("--out", default="./charge_eval_out")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ev-type", default="electric_vehicle")
    p.add_argument("--sumo-binary", default="sumo")
    p.add_argument("--max-steps", type=int, default=100000)
    p.add_argument("--sample-every", type=int, default=1, help="steps between SoC samples")
    p.add_argument("--soc-mean", type=float, default=50.0, help="initial SoC mean, %%")
    p.add_argument("--soc-sd", type=float, default=10.0)
    p.add_argument("--wh-per-km", type=float, default=170.0,
                   help="planner's energy proxy; compare with sim_wh_per_km in the summary")
    p.add_argument("--depleted-soc", type=float, default=0.5,
                   help="a trip counts as depleted if SoC ever drops to this %% or below")
    p.add_argument("--risk-mean", type=float, default=0.5)
    p.add_argument("--risk-sd", type=float, default=0.2)
    p.add_argument("--llm-endpoint", default="http://localhost:8000/v1")
    p.add_argument("--llm-endpoint-file", default=None,
                   help="endpoint JSON written by llm_server/serve_model.slurm, or a directory "
                        "of them (newest live one is used); overrides --llm-endpoint")
    p.add_argument("--llm-concurrency", type=int, default=16,
                   help="max simultaneous LLM requests (vehicles departing in the same step)")
    p.add_argument("--llm-model", default=None, help="defaults to the first model the server lists")
    p.add_argument("--llm-thinking", action="store_true", help="leave Qwen3 thinking mode on")
    args = p.parse_args()

    if args.llm_endpoint_file and "llm" in args.policies:
        args.llm_endpoint = charge_policies.endpoint_from_file(args.llm_endpoint_file)
        print(f"Using LLM endpoint {args.llm_endpoint}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for name in args.policies:
        out_dir = os.path.join(args.out, name)
        policy = charge_policies.make_policy(name, args, out_dir if name != "llm" else args.out)
        print(f"=== running {name} ===", flush=True)
        rows.append(run_policy(policy, args, out_dir))

    write_csv(os.path.join(args.out, "summary.csv"), rows)
    width = max(len(k) for k in rows[0])
    print("\n" + "metric".ljust(width) + "".join(r['policy'].rjust(12) for r in rows))
    for key in rows[0]:
        if key != 'policy':
            print(key.ljust(width) + "".join(str(r[key]).rjust(12) for r in rows))
    print(f"\nWrote {os.path.join(args.out, 'summary.csv')}")


if __name__ == "__main__":
    main()
