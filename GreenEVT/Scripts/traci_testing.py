import csv
import os
import random
import sys
from pathlib import Path

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci


SCRIPT_DIR = Path(__file__).resolve().parent

HOME_CHARGE_POWER_W = 7200.0
HOME_CHARGE_EFFICIENCY = 0.95
SIMULATION_DAYS = 1


def _resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else SCRIPT_DIR / path


def activitygen_vehicle_key(vehicle_id):
    """Map ActivityGen trip IDs to the shared person/household ID."""
    return vehicle_id.rsplit(":", 1)[0]


def _trip_number(vehicle_id):
    """Read the ActivityGen trip number from IDs such as ``h277c1:3``."""
    try:
        return int(vehicle_id.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return None


def _remember_home_edge(vehicle_id, route, home_edges):
    """Learn a household's home edge from ActivityGen's vehicle routes."""
    if not route:
        return

    household = activitygen_vehicle_key(vehicle_id)
    if _trip_number(vehicle_id) == 1:
        home_edges[household] = route[0]
    else:
        home_edges.setdefault(household, route[0])


def _is_home_return(vehicle_id, route, home_edges, home_return_trip_numbers):
    """Identify a return-home trip without using traffic stops as a signal."""
    if not route:
        return False

    household = activitygen_vehicle_key(vehicle_id)
    home_edge = home_edges.get(household)
    if home_edge is not None:
        return home_edge == route[-1]

    # This is only a compatibility fallback for demand files that do not let
    # us learn a household's home edge from an earlier trip. ActivityGen's
    # trip numbers change with the number of activities and across days, so a
    # single default trip number cannot identify all return-home trips.
    if home_return_trip_numbers is None:
        return False

    if _trip_number(vehicle_id) in set(home_return_trip_numbers):
        return True
    return False


def _route_stage(from_edge, to_edge, vtype="electric_vehicle"):
    try:
        # stage = traci.simulation.findRoute(from_edge, to_edge)
        stage = traci.simulation.findRoute(from_edge, to_edge, vType=vtype,
                                           depart=-1,
                                           routingMode=0)
    except traci.exceptions.TraCIException as e:
        print(f"traci exception {e}")
        return None

    return stage if stage.edges else None


def _distance_from_vehicle(edge_id, position, destination_edge, fallback):
    try:
        distance = traci.simulation.getDistanceRoad(
            edge_id,
            position,
            destination_edge,
            0.0,
            isDriving=True,
        )
        return distance if distance >= 0 else fallback
    except traci.exceptions.TraCIException:
        return fallback


def choose_least_impact_charger(
    vehicle_id,
    destination_edge,
    current_charge,
    battery_capacity,
    charging_stations,
    reserve_soc=0.10,
    estimated_consumption_wh_per_meter=0.20,
    charge_target_soc=0.90,
):
    """Choose a charger by added travel time, then added route distance.

    The decision is also made here: ``None`` means the vehicle has enough
    estimated energy to reach its destination with the requested reserve.
    ``estimated_consumption_wh_per_meter`` should be calibrated to the SUMO
    battery model used by the route file.
    """
    current_edge = traci.vehicle.getRoadID(vehicle_id)
    current_position = traci.vehicle.getLanePosition(vehicle_id)
    baseline = _route_stage(current_edge, destination_edge)
    if baseline is None:
        return None

    baseline_distance = _distance_from_vehicle(
        current_edge,
        current_position,
        destination_edge,
        baseline.length,
    )
    reserve_energy = battery_capacity * reserve_soc
    required_energy = (
        baseline_distance * estimated_consumption_wh_per_meter + reserve_energy
    )

    if current_charge >= required_energy:
        return None

    best = None
    for station_id in charging_stations:
        station_lane = traci.chargingstation.getLaneID(station_id)
        station_edge = traci.lane.getEdgeID(station_lane)

        to_station = _route_stage(current_edge, station_edge)
        from_station = _route_stage(station_edge, destination_edge)
        if to_station is None or from_station is None:
            continue

        distance_to_station = _distance_from_vehicle(
            current_edge,
            current_position,
            station_edge,
            to_station.length,
        )
        if current_charge < (
            distance_to_station * estimated_consumption_wh_per_meter
            + reserve_energy
        ):
            continue

        candidate_distance = distance_to_station + from_station.length
        candidate_time = to_station.travelTime + from_station.travelTime
        added_distance = max(0.0, candidate_distance - baseline_distance)
        added_time = max(0.0, candidate_time - baseline.travelTime)

        target_charge = max(
            battery_capacity * charge_target_soc,
            from_station.length * estimated_consumption_wh_per_meter
            + reserve_energy,
        )
        target_charge = min(battery_capacity, target_charge)

        candidate = {
            "station": station_id,
            "edge": station_edge,
            "added_time": added_time,
            "added_distance": added_distance,
            "target_charge": target_charge,
        }
        score = (added_time, added_distance, candidate_time, station_id)

        if best is None or score < best["score"]:
            candidate["score"] = score
            best = candidate

    return best


# findRoute builds a throwaway vehicle of the given type, which draws a speed
# factor from SUMO's random number generator. That shifts every later
# vehicle's random draws, so a policy that routes more (or at different
# times) would change traffic for reasons unrelated to its decisions.
# decision_state routes with a copy of the EV type that has no speed
# deviation: same travel times, no random draws.
ROUTER_VTYPE = "charge_eval_router"


def _ensure_router_vtype():
    if ROUTER_VTYPE not in traci.vehicletype.getIDList():
        traci.vehicletype.copy("electric_vehicle", ROUTER_VTYPE)
        traci.vehicletype.setSpeedDeviation(ROUTER_VTYPE, 0.0)


def decision_state(ctx):
    """Everything a charge decision can depend on, measured once per leg.

    Unlike ``choose_least_impact_charger`` this routes to *every* charger
    (no reserve filter, no early exit), so the result is independent of the
    policy and any policy can decide from it -- live, or later when replaying
    a logged run (see charge_policies.py and replay_decisions.py). Plain
    JSON-serialisable values only.
    """
    _ensure_router_vtype()
    vehicle_id = ctx["vehicle_id"]
    destination_edge = ctx["destination_edge"]
    current_edge = traci.vehicle.getRoadID(vehicle_id)
    current_position = traci.vehicle.getLanePosition(vehicle_id)
    state = {key: ctx[key] for key in (
        "vehicle_id", "household", "destination_edge", "current_charge", "capacity",
        "reserve_soc", "estimated_consumption_wh_per_meter", "charge_target_soc",
        "current_time", "is_home_return", "home_charge_power_w")}
    state.update({"current_edge": current_edge, "current_position": current_position,
                  "baseline": None, "stations": []})

    baseline = _route_stage(current_edge, destination_edge, ROUTER_VTYPE)
    if baseline is None:
        return state
    state["baseline"] = {
        "travel_time": baseline.travelTime,
        "length": baseline.length,
        "distance": _distance_from_vehicle(
            current_edge, current_position, destination_edge, baseline.length),
    }

    for station_id in ctx["charging_stations"]:
        station_lane = traci.chargingstation.getLaneID(station_id)
        station_edge = traci.lane.getEdgeID(station_lane)

        to_station = _route_stage(current_edge, station_edge, ROUTER_VTYPE)
        from_station = _route_stage(station_edge, destination_edge, ROUTER_VTYPE)
        if to_station is None or from_station is None:
            continue

        try:
            power_w = traci.chargingstation.getChargingPower(station_id)
        except (AttributeError, traci.exceptions.TraCIException):
            power_w = None
        state["stations"].append({
            "station": station_id,
            "edge": station_edge,
            "to_travel_time": to_station.travelTime,
            "to_length": to_station.length,
            "distance_to_station": _distance_from_vehicle(
                current_edge, current_position, station_edge, to_station.length),
            "from_travel_time": from_station.travelTime,
            "from_length": from_station.length,
            "power_w": power_w,
            "vehicles_at_station": traci.chargingstation.getVehicleCount(station_id),
        })
    return state


def _least_impact_policy(ctx):
    """Default charge decision: the team's least-impact rule, unchanged."""
    choice = choose_least_impact_charger(
        ctx["vehicle_id"],
        ctx["destination_edge"],
        ctx["current_charge"],
        ctx["capacity"],
        ctx["charging_stations"],
        reserve_soc=ctx["reserve_soc"],
        estimated_consumption_wh_per_meter=ctx["estimated_consumption_wh_per_meter"],
        charge_target_soc=ctx["charge_target_soc"],
    )
    return lambda: (choice, {})


def _remember_battery_levels(battery_state):
    """Save live battery levels before ActivityGen removes a trip vehicle."""
    for vehicle_id in traci.vehicle.getIDList():
        if traci.vehicle.getTypeID(vehicle_id) != "electric_vehicle":
            continue

        key = activitygen_vehicle_key(vehicle_id)
        try:
            battery_state[key] = float(
                traci.vehicle.getParameter(
                    vehicle_id,
                    "device.battery.chargeLevel",
                )
            )
        except traci.exceptions.TraCIException:
            continue


def _default_initial_charge(key, capacity):
    return random.gauss(0.5 * capacity, 0.1 * capacity)


def _initialize_battery(
    vehicle_id,
    battery_state,
    home_state,
    current_time,
    home_charge_power_w,
    home_charge_efficiency,
    initial_charge_fn=_default_initial_charge,
    verbose=True,
):
    capacity = float(
        traci.vehicle.getParameter(vehicle_id, "device.battery.capacity")
    )
    key = activitygen_vehicle_key(vehicle_id)
    charge = battery_state.get(key)

    if charge is None:
        charge = initial_charge_fn(key, capacity)

    home_arrival = home_state.pop(key, None)
    home_energy_added = 0.0
    if home_arrival is not None:
        dwell_seconds = max(0.0, current_time - home_arrival["arrival_time"])
        requested_energy = (
            home_charge_power_w
            * dwell_seconds
            / 3600.0
            * home_charge_efficiency
        )
        charge = home_arrival["charge"] + requested_energy
        if verbose:
            print(
                f"{vehicle_id}: home charged {requested_energy:.1f} Wh "
                f"after {dwell_seconds / 3600.0:.2f} h"
            )

    charge = max(0.0, min(capacity, charge))
    if home_arrival is not None:
        home_energy_added = max(0.0, charge - home_arrival["charge"])
    traci.vehicle.setParameter(
        vehicle_id,
        "device.battery.chargeLevel",
        str(charge),
    )
    battery_state[key] = charge
    return capacity, charge, home_energy_added


def _write_charging_totals(output_file, station_energy_wh, home_energy_wh):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    total_energy_wh = sum(station_energy_wh.values()) + home_energy_wh
    with output_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("charging_source", "station_id",
                        "energy_wh", "energy_kwh"))
        for station_id, energy_wh in sorted(station_energy_wh.items()):
            writer.writerow(
                ("station", station_id, f"{energy_wh:.2f}", f"{energy_wh / 1000:.5f}"))
        writer.writerow(
            ("home", "", f"{home_energy_wh:.2f}", f"{home_energy_wh / 1000:.5f}"))
        writer.writerow(
            ("total", "", f"{total_energy_wh:.2f}", f"{total_energy_wh / 1000:.5f}"))


def _prepare_dispatch(
    vehicle_id,
    vehicle_info,
    battery_state,
    home_state,
    home_charging_total,
    current_time,
    home_edges,
    home_return_trip_numbers,
    charging_stations,
    reserve_soc,
    estimated_consumption_wh_per_meter,
    charge_target_soc,
    home_charge_power_w,
    home_charge_efficiency,
    initial_charge_fn,
    verbose,
    home_sessions=None,
):
    """Set up the vehicle's current leg; returns the decision context, or None."""
    if traci.vehicle.getTypeID(vehicle_id) != "electric_vehicle":
        return None

    route = traci.vehicle.getRoute(vehicle_id)
    if not route:
        return None

    household = activitygen_vehicle_key(vehicle_id)
    _remember_home_edge(vehicle_id, route, home_edges)

    home_arrival = home_state.get(household)
    capacity, charge, home_energy_added = _initialize_battery(
        vehicle_id,
        battery_state,
        home_state,
        current_time,
        home_charge_power_w,
        home_charge_efficiency,
        initial_charge_fn,
        verbose,
    )
    home_charging_total["energy_wh"] += home_energy_added
    if home_sessions is not None and home_arrival is not None and home_energy_added > 0:
        # Charging is applied as a lump sum at departure; for demand curves,
        # place it from arrival home at the home charger's rate.
        hours = home_energy_added / (home_charge_power_w * home_charge_efficiency)
        home_sessions.append({
            "household": household,
            "start_s": home_arrival["arrival_time"],
            "end_s": home_arrival["arrival_time"] + hours * 3600.0,
            "energy_wh": home_energy_added,
            "power_w": home_charge_power_w,
        })
    destination_edge = route[-1]

    info = vehicle_info.setdefault(vehicle_id, {})
    info.update({
        "activitygen_key": household,
        "destination": destination_edge,
        "is_home_return": _is_home_return(
            vehicle_id,
            route,
            home_edges,
            home_return_trip_numbers,
        ),
        "state": "to_destination",
    })

    return {
        "vehicle_id": vehicle_id,
        "household": household,
        "destination_edge": destination_edge,
        "current_charge": charge,
        "capacity": capacity,
        "charging_stations": charging_stations,
        "reserve_soc": reserve_soc,
        "estimated_consumption_wh_per_meter": estimated_consumption_wh_per_meter,
        "charge_target_soc": charge_target_soc,
        "current_time": current_time,
        "is_home_return": info["is_home_return"],
        "home_charge_power_w": home_charge_power_w,
    }


def _apply_choice(vehicle_id, info, choice, verbose):
    """Send the vehicle to its chosen charger (``choice`` None = drive on)."""
    if choice is None:
        return

    info.update({
        "station": choice["station"],
        "station_edge": choice["edge"],
        "target_charge": choice["target_charge"],
        "state": "to_charger",
    })

    if verbose:
        print(f"before change target : {vehicle_id} , {choice['edge']}")

    traci.vehicle.changeTarget(vehicle_id, choice["edge"])
    if verbose:
        print(f"after change target : {vehicle_id} , {choice['edge']}")
        temp_r = traci.vehicle.getRoute(vehicle_id)
        print(f"vechicle route : {temp_r}")
        print(f"station edge : {choice['edge']}")
        print(f"choice station: {choice['station']}")
        print(f"choice station in temp r {choice['edge'] in temp_r}")

    traci.vehicle.setChargingStationStop(
        vehicle_id,
        choice["station"],
        duration=100000.0,
    )

    if verbose:
        print(
            f"{vehicle_id}: charging at {choice['station']} "
            f"(added time={choice['added_time']:.1f}s, "
            f"added distance={choice['added_distance']:.1f}m)"
        )


def _close_leg(open_legs, trip_log, vehicle_id, end, current_time, charge=None):
    leg = open_legs.pop(vehicle_id, None)
    if leg is None:
        return
    leg["end"] = end
    leg["arrive_s"] = current_time
    leg["trip_min"] = round((current_time - leg["depart_s"]) / 60.0, 2)
    if charge is not None:
        leg["arrival_soc"] = round(charge / leg["capacity_wh"] * 100.0, 2)
    trip_log.append(leg)


def run_traci_simulation(
    tracker_file=None,
    route_file=".\\test_PA_rou.xml",
    net_file="..\\data\\Palo_Alto\\PA.network.net.xml",
    reserve_soc=0.10,
    estimated_consumption_wh_per_meter=0.20,
    charge_target_soc=0.90,
    home_return_trip_numbers=(),
    home_charge_power_w=HOME_CHARGE_POWER_W,
    home_charge_efficiency=HOME_CHARGE_EFFICIENCY,
    simulation_days=SIMULATION_DAYS,
    charging_totals_file=None,
    charge_policy=None,
    chargers_file=None,
    initial_charge_fn=_default_initial_charge,
    trip_log=None,
    decision_log=None,
    home_sessions=None,
    station_demand=None,
    demand_bin_s=60,
    sumo_seed=None,
    sumo_log="./logging.log",
    verbose=True,
):
    """Run ActivityGen trips and make charging decisions through TraCI.

    ActivityGen remains responsible for home/work routing. TraCI only changes
    the current leg when charging is needed, then restores the ActivityGen
    destination after charging. The ActivityGen schedule is expected to have
    been generated with the same ``simulation_days`` value (five by default).

    Optional hooks used by charge_eval.py (defaults keep the behavior above):
    ``charge_policy`` replaces the least-impact rule (see charge_policies.py)
    and decides from ``decision_state()``; ``trip_log`` (a list) receives one
    dict per EV leg; ``decision_log`` (a list) receives the full state and
    choice of every decision, for replaying other policies on the same states;
    ``home_sessions`` (a list) and ``station_demand`` (a dict keyed by
    (bin start s, station)) record charging demand over time;
    ``initial_charge_fn`` sets a household's first battery level;
    ``verbose=False`` silences the per-step prints. Vehicles are handled in
    sorted ID order each step so runs don't depend on Python's hash seed.
    """
    if not isinstance(simulation_days, int) or simulation_days <= 0:
        raise ValueError("simulation_days must be a positive integer")

    route_file = _resolve_path(route_file)
    net_file = _resolve_path(net_file)
    if charging_totals_file is None:
        report_directory = (
            route_file.parent.parent / "outputs"
            if route_file.parent.name == "routes"
            else route_file.parent
        )
        charging_totals_file = report_directory / (
            f"{route_file.stem}_charging_totals.csv"
        )
    else:
        charging_totals_file = _resolve_path(charging_totals_file)
    additional_files = [_resolve_path(chargers_file) if chargers_file
                        else SCRIPT_DIR / "chargers_additional.xml"]
    if tracker_file is not None:
        additional_files.append(_resolve_path(tracker_file))

    traci_cmd = [
        "sumo",
        "--net-file", str(net_file),
        "--route-files", str(route_file),
        "--additional-files", ",".join(str(path) for path in additional_files),
        "--time-to-teleport", "300",
        "--end", str(simulation_days * 24 * 60 * 60),
        # "--no-warnings", "true",
        "--log", str(sumo_log),
    ]
    if sumo_seed is not None:
        # Fixed seed and single-threaded routing: identical runs every time.
        traci_cmd += ["--seed", str(sumo_seed), "--threads", "1"]
    if not verbose:
        traci_cmd += ["--no-step-log", "true"]

    if charge_policy is None:
        choose, resolve = _least_impact_policy, None
    else:
        choose = charge_policy.choose
        resolve = getattr(charge_policy, "resolve", None)

    traci.start(traci_cmd)
    charging_stations = traci.chargingstation.getIDList()
    station_energy_wh = {station_id: 0.0 for station_id in charging_stations}
    home_charging_total = {"energy_wh": 0.0}
    vehicle_info = {}
    battery_state = {}
    home_state = {}
    home_edges = {}
    open_legs = {}  # vehicle_id -> trip_log record for the leg in progress
    current_time = 0.0

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            previous_vehicle_charge = {}
            for vehicle_id in vehicle_info:
                try:
                    previous_vehicle_charge[vehicle_id] = float(
                        traci.vehicle.getParameter(
                            vehicle_id,
                            "device.battery.chargeLevel",
                        )
                    )
                except traci.exceptions.TraCIException:
                    continue

            _remember_battery_levels(battery_state)
            if verbose:
                print(
                    "before main step",
                    traci.simulation.getTime(),
                    flush=True,
                )
            traci.simulationStep()
            if verbose:
                print(
                    "after main step",
                    traci.simulation.getTime(),
                    flush=True,
                )
            current_time = traci.simulation.getTime()

            vehicles_at_stations = {}
            for station_id in charging_stations:
                for vehicle_id in traci.chargingstation.getVehicleIDs(station_id):
                    vehicles_at_stations[vehicle_id] = station_id

            for vehicle_id, info in vehicle_info.items():
                station_id = vehicles_at_stations.get(vehicle_id)
                if info["state"] != "charging" and station_id is None:
                    continue
                previous_charge = previous_vehicle_charge.get(vehicle_id)
                if previous_charge is None:
                    continue
                try:
                    current_charge = float(
                        traci.vehicle.getParameter(
                            vehicle_id,
                            "device.battery.chargeLevel",
                        )
                    )
                except traci.exceptions.TraCIException:
                    continue
                energy_added = max(0.0, current_charge - previous_charge)
                if station_id is None:
                    station_id = info.get("station")
                if station_id in station_energy_wh:
                    station_energy_wh[station_id] += energy_added
                    if station_demand is not None and energy_added > 0:
                        bin_start = int(current_time // demand_bin_s) * demand_bin_s
                        key = (bin_start, station_id)
                        station_demand[key] = station_demand.get(key, 0.0) + energy_added

            departed = set(traci.simulation.getDepartedIDList())
            arrived = set(traci.simulation.getArrivedIDList())
            stop_ended = set(
                traci.simulation.getStopEndingVehiclesIDList()
            )

            # SUMO removes a vehicle at the end of its route.  Save the latest
            # SOC so the next trip for that household can receive the energy
            # accumulated while it was at home.
            for vehicle_id in sorted(arrived):
                info = vehicle_info.pop(vehicle_id, None)
                if trip_log is not None and info is not None:
                    _close_leg(open_legs, trip_log, vehicle_id, "arrived", current_time,
                               battery_state.get(info["activitygen_key"]))
                if info is None or not info.get("is_home_return"):
                    continue

                charge = battery_state.get(info["activitygen_key"])
                if charge is not None:
                    home_state[info["activitygen_key"]] = {
                        "charge": charge,
                        "arrival_time": current_time,
                    }

            dispatch_args = (
                vehicle_info,
                battery_state,
                home_state,
                home_charging_total,
                current_time,
                home_edges,
                home_return_trip_numbers,
                charging_stations,
                reserve_soc,
                estimated_consumption_wh_per_meter,
                charge_target_soc,
                home_charge_power_w,
                home_charge_efficiency,
                initial_charge_fn,
                verbose,
                home_sessions,
            )
            contexts = []

            # ActivityGen normally creates a new vehicle ID for each trip.
            # This catches both the home and work legs.
            for vehicle_id in sorted(departed):
                route = traci.vehicle.getRoute(vehicle_id)

                # Learn the home edge from every ActivityGen vehicle, not
                # only EVs. EVs are assigned per trip, so an EV may first
                # appear on trip 2 or 3 for a household.
                _remember_home_edge(vehicle_id, route, home_edges)

                contexts.append(_prepare_dispatch(vehicle_id, *dispatch_args))

            # If ActivityGen represents work as a stop on the same vehicle,
            # the vehicle is leaving through this event rather than a new
            # departed-ID event.
            for vehicle_id in sorted(stop_ended):
                info = vehicle_info.get(vehicle_id)
                if info is None or info.pop("ignore_stop_end", False):
                    continue
                if trip_log is not None:
                    _close_leg(open_legs, trip_log, vehicle_id, "stop", current_time,
                               battery_state.get(info["activitygen_key"]))

                contexts.append(_prepare_dispatch(vehicle_id, *dispatch_args))

            # Decide every leg starting this step before acting on any, so a
            # policy can run its decisions concurrently (e.g. LLM requests).
            contexts = [ctx for ctx in contexts if ctx is not None]
            states = ([decision_state(ctx) for ctx in contexts]
                      if charge_policy is not None else contexts)
            finishers = [choose(state) for state in states]
            decisions = resolve(finishers) if resolve else [finish() for finish in finishers]
            for ctx, state, (choice, decision_info) in zip(contexts, states, decisions):
                vehicle_id = ctx["vehicle_id"]
                if decision_log is not None:
                    decision_log.append({
                        "state": state,
                        "choice": None if choice is None else {
                            key: choice[key] for key in ("station", "edge", "target_charge")},
                        "info": decision_info,
                    })
                if trip_log is not None:
                    open_legs[vehicle_id] = {
                        "vehicle_id": vehicle_id,
                        "household": ctx["household"],
                        "depart_s": current_time,
                        "capacity_wh": ctx["capacity"],
                        "start_soc": round(ctx["current_charge"] / ctx["capacity"] * 100.0, 2),
                        "min_soc": ctx["current_charge"] / ctx["capacity"] * 100.0,
                        "is_home_return": vehicle_info[vehicle_id]["is_home_return"],
                        "station": choice["station"] if choice else "",
                        "target_soc": round(choice["target_charge"] / ctx["capacity"] * 100.0, 2) if choice else "",
                        "charge_stops": 0, "charge_min": 0.0, "wait_min": 0.0, "charged_kwh": 0.0,
                        **decision_info,
                    }
                _apply_choice(vehicle_id, vehicle_info[vehicle_id], choice, verbose)

            active_vehicle_ids = set(traci.vehicle.getIDList())
            for vehicle_id, info in list(vehicle_info.items()):
                if vehicle_id not in active_vehicle_ids:
                    vehicle_info.pop(vehicle_id)
                    if trip_log is not None:
                        _close_leg(open_legs, trip_log, vehicle_id, "vanished", current_time,
                                   battery_state.get(info["activitygen_key"]))
                    continue

                if info["state"] == "to_charger":
                    station_vehicles = set(
                        traci.chargingstation.getVehicleIDs(info["station"])
                    )
                    if vehicle_id in station_vehicles:
                        info["state"] = "charging"
                        if vehicle_id in open_legs:
                            open_legs[vehicle_id]["_charge_start"] = (
                                current_time, battery_state.get(info["activitygen_key"], 0.0))
                    elif (vehicle_id in open_legs
                          and traci.vehicle.getRoadID(vehicle_id) == info["station_edge"]
                          and traci.vehicle.getSpeed(vehicle_id) < 0.1):
                        # Queued behind the station's occupied stopping space.
                        open_legs[vehicle_id]["wait_min"] += traci.simulation.getDeltaT() / 60.0

                elif info["state"] == "charging":
                    charge = float(
                        traci.vehicle.getParameter(
                            vehicle_id,
                            "device.battery.chargeLevel",
                        )
                    )
                    if verbose:
                        print(
                            "charging",
                            vehicle_id,
                            "station=", info["station"],
                            "charge=", charge,
                            "target=", info["target_charge"],
                            "station vehicles=",
                            traci.chargingstation.getVehicleIDs(info["station"]),
                            "time=", traci.simulation.getTime(),
                            flush=True,
                        )
                    if charge >= info["target_charge"]:
                        if verbose:
                            print(
                                "resuming",
                                vehicle_id,
                                "station=", info["station"],
                                "destination=", info["destination"],
                                "charge=", charge,
                                "target=", info["target_charge"],
                                flush=True,
                            )
                        traci.vehicle.resume(vehicle_id)
                        traci.vehicle.changeTarget(
                            vehicle_id,
                            info["destination"],
                        )
                        info["ignore_stop_end"] = True
                        info["state"] = "to_destination"
                        leg = open_legs.get(vehicle_id)
                        if leg is not None and "_charge_start" in leg:
                            start_time, start_charge = leg.pop("_charge_start")
                            leg["charge_stops"] += 1
                            leg["charge_min"] += (current_time - start_time) / 60.0
                            leg["charged_kwh"] += max(0.0, charge - start_charge) / 1000.0

            _remember_battery_levels(battery_state)
            for vehicle_id, leg in open_legs.items():
                charge = battery_state.get(leg["household"])
                if charge is not None:
                    leg["min_soc"] = min(leg["min_soc"], charge / leg["capacity_wh"] * 100.0)
    finally:
        try:
            traci.close()
        finally:
            if trip_log is not None:
                for vehicle_id in list(open_legs):
                    _close_leg(open_legs, trip_log, vehicle_id, "unfinished", current_time)
            _write_charging_totals(
                charging_totals_file,
                station_energy_wh,
                home_charging_total["energy_wh"],
            )
            print(f"Charging totals saved to {charging_totals_file}")
    return {
        "station_energy_wh": station_energy_wh,
        "home_energy_wh": home_charging_total["energy_wh"],
        # Households home at the end that never left again: their overnight
        # charging never happens in the sim, but belongs in a demand curve.
        "home_pending": [
            {"household": key, "arrival_s": value["arrival_time"], "charge_wh": value["charge"]}
            for key, value in sorted(home_state.items())
        ],
    }


if __name__ == "__main__":
    run_traci_simulation(
        route_file="..\\genetic_alg\\pa_test\\intermediate_files\\routes\\2_34_rou.xml",
        tracker_file="..\\genetic_alg\\pa_test\\intermediate_files\\trackers\\2_34_tracker.xml"
    )
