import heapq
import itertools
import warnings

import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci
import random


HOME_CHARGE_POWER_W = 7200.0
HOME_CHARGE_EFFICIENCY = 0.95
SIMULATION_DAYS = 5


def _activitygen_vehicle_key(vehicle_id):
    """Return the household/person key shared by ActivityGen trip vehicles."""
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

    household = _activitygen_vehicle_key(vehicle_id)
    if _trip_number(vehicle_id) == 1:
        home_edges[household] = route[0]
    else:
        home_edges.setdefault(household, route[0])


def _remember_battery_levels(battery_state):
    """Save live EV battery levels before SUMO removes finished vehicles."""
    for vehicle_id in traci.vehicle.getIDList():
        if traci.vehicle.getTypeID(vehicle_id) != "electric_vehicle":
            continue

        try:
            battery_state[_activitygen_vehicle_key(vehicle_id)] = float(
                traci.vehicle.getParameter(
                    vehicle_id,
                    "device.battery.chargeLevel",
                )
            )
        except traci.exceptions.TraCIException:
            # The vehicle may not have a battery device despite its type ID.
            continue


def _initialize_battery(
    vehicle_id,
    battery_state,
    home_state,
    current_time,
    home_charge_power_w,
    home_charge_efficiency,
):
    """Initialize a new trip vehicle and apply charging done while at home."""
    key = _activitygen_vehicle_key(vehicle_id)
    capacity = float(
        traci.vehicle.getParameter(vehicle_id, "device.battery.capacity")
    )
    charge = battery_state.get(key)

    if charge is None:
        charge = random.gauss(0.5 * capacity, 0.1 * capacity)

    home_arrival = home_state.pop(key, None)
    if home_arrival is not None:
        dwell_seconds = max(0.0, current_time - home_arrival["arrival_time"])
        energy_added = (
            home_charge_power_w
            * dwell_seconds
            / 3600.0
            * home_charge_efficiency
        )
        charge = home_arrival["charge"] + energy_added
        print(
            f"{vehicle_id}: home charged {energy_added:.1f} Wh "
            f"after {dwell_seconds / 3600.0:.2f} h"
        )

    charge = max(0.0, min(capacity, charge))
    traci.vehicle.setParameter(
        vehicle_id,
        "device.battery.chargeLevel",
        str(charge),
    )
    battery_state[key] = charge


def _is_home_return(vehicle_id, route, home_edges, home_return_trip_numbers):
    """Identify a return-home trip without treating traffic stops as arrivals."""
    if not route:
        return False

    key = _activitygen_vehicle_key(vehicle_id)
    home_edge = home_edges.get(key)
    if home_edge is not None:
        return home_edge == route[-1]

    if home_return_trip_numbers is None:
        return False

    if _trip_number(vehicle_id) in set(home_return_trip_numbers):
        return True
    return False


def run_traci_simulation(
    tracker_file=None,
    route_file=".\\test_PA_rou.xml",
    net_file="..\\data\\Palo_Alto\\PA.network.net.xml",
    home_return_trip_numbers=(),
    home_charge_power_w=HOME_CHARGE_POWER_W,
    home_charge_efficiency=HOME_CHARGE_EFFICIENCY,
    simulation_days=SIMULATION_DAYS,
):
    """Run SUMO and carry EV battery state between ActivityGen trips.

    ActivityGen commonly creates IDs like ``h277c1:1`` and ``h277c1:3``.
    Trip numbers are not reliable return-home markers because they vary with
    the activity pattern and reset by offset on each generated day. A
    household's home edge is learned from its first observed trip, and
    charging is applied at the next departure using the time spent between
    trips.

    ActivityGen trip numbers vary with the number of activities and across
    days. Home edges are learned from the first observed vehicle trip for each
    household, so route destinations identify return-home trips consistently
    across the five-day schedule. ``home_return_trip_numbers`` remains an
    optional compatibility fallback for custom demand files.
    """
    if not isinstance(simulation_days, int) or simulation_days <= 0:
        raise ValueError("simulation_days must be a positive integer")

    additional_files = f".\\chargers_additional.xml"
    if tracker_file is not None:
        additional_files += f",{tracker_file}"

    traci_cmd = [
        "sumo",
        "--net-file", net_file,
        "--route-files", route_file,
        "--additional-files", additional_files,
        "--chargingstations-output", ".\\charging_stations_output.xml",
        "--time-to-teleport", "-1",
        "--end", str(simulation_days * 24 * 60 * 60),
        "--no-warnings", "true",
    ]

    traci.start(traci_cmd)

    vehicle_info = {}
    battery_state = {}
    home_state = {}
    home_edges = {}

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            # Vehicles that arrive this step are removed by SUMO.  Capture
            # their latest battery level before advancing the simulation.
            _remember_battery_levels(battery_state)
            traci.simulationStep()
            current_time = traci.simulation.getTime()

            arrived = set(traci.simulation.getArrivedIDList())
            for vehicle_id in arrived:
                info = vehicle_info.pop(vehicle_id, None)
                if info is None or not info["is_home_return"]:
                    continue

                charge = battery_state.get(info["household"])
                if charge is not None:
                    home_state[info["household"]] = {
                        "charge": charge,
                        "arrival_time": current_time,
                    }

            for vehicle_id in traci.simulation.getDepartedIDList():
                route = traci.vehicle.getRoute(vehicle_id)
                # Home-edge discovery must include non-EVs because EVs are
                # selected independently for each ActivityGen trip.
                _remember_home_edge(vehicle_id, route, home_edges)

                if traci.vehicle.getTypeID(vehicle_id) != "electric_vehicle":
                    continue

                household = _activitygen_vehicle_key(vehicle_id)
                _initialize_battery(
                    vehicle_id,
                    battery_state,
                    home_state,
                    current_time,
                    home_charge_power_w,
                    home_charge_efficiency,
                )
                vehicle_info[vehicle_id] = {
                    "household": household,
                    "is_home_return": _is_home_return(
                        vehicle_id,
                        route,
                        home_edges,
                        home_return_trip_numbers,
                    ),
                }

            _remember_battery_levels(battery_state)
    finally:
        traci.close()



# Add minimum charge level when leaving chargin station

def _estimate_wait(status_entry, default_session_minutes=30.0):
    if status_entry is None:
        return 0.0
    if isinstance(status_entry, (int, float)):
        return float(status_entry)

    stalls = status_entry['stalls']
    occupied = status_entry.get('occupied', 0)
    avg_session = status_entry.get('avg_session_minutes', default_session_minutes)

    if stalls <= 0:
        raise ValueError(f"station capacity 'stalls' must be positive, got {stalls}")
    if occupied < stalls:
        return 0.0

    cars_ahead_of_us = occupied - stalls + 1
    return (cars_ahead_of_us / stalls) * avg_session

def plan_ev_route(origin, destination, initial_soc, stations_graph, vehicle_profile,
                   real_time_status, soc_step=10, min_destination_soc=0.0):

    # NOTE: min_destination_soc defaults to 0.0 intentionally during this testing phase. 
    # This prevents tight mathematical constraints from returning unviable routes, maximizing 
    # the volume of actionable route plans generated to stress-test the simulation's traci execution loop.
    
    """
    Plans a minimum-time EV route from origin to destination, choosing both which
    roads to drive and how much to charge at each station along the way, using a
    state-expanded Dijkstra search over (location, state_of_charge) pairs.

    Note on optimality: this finds the true minimum-time route given the constraint
    that charging can only target multiples of soc_step (plus 100%, always offered
    explicitly). States are pruned via SoC-dominance, not bucketing/rounding, so
    there's no approximation error beyond that grid constraint itself -- lowering
    soc_step gets you a finer charge-target grid, not a "more correct" search.

    Args:
        origin (str): Starting node name. Never treated as a charging location,
            regardless of what's in stations_graph -- the vehicle's departure
            charge is assumed to already be decided by the caller before this
            function runs (e.g. "left home at 100%"), not something this search
            optimizes over.

        destination (str): Target node name. The search terminates the moment
            this node is reached, so it's likewise never treated as a place to
            charge.

        initial_soc (float): Starting battery state of charge as a percentage
            (0-100), as of right now -- not as of some future planned top-up.

        stations_graph (dict): Adjacency list describing the road network:
                {
                    node_name: {
                        neighbor_name: {
                            'driving_time': float,  # minutes to drive this edge
                            'soc_cost': float        # % battery used driving it
                        },
                        ...
                    },
                    ...
                }
            Every node the route might touch (including destination) should
            appear as a key, even if just with an empty neighbor dict.

        vehicle_profile (dict): Vehicle-specific parameters. Currently just:
                {'charge_curve_func': callable(from_soc, to_soc) -> minutes}
            Swap this function out to model different vehicles or charger
            hardware; the search itself doesn't assume any particular curve
            shape.

        real_time_status (dict): {station_name: status}. A SNAPSHOT taken at
            planning time, not a time-varying forecast -- this is intentional
            (see discussion), so the result is optimal for conditions *right
            now*, not necessarily conditions at actual arrival time.
            status is one of:
              - a dict {'stalls': int, 'occupied': int,
                        'avg_session_minutes': float (optional, default 30)}
                -- wait is estimated from capacity (0 if occupied < stalls,
                otherwise scaled by how many cars are ahead per stall).
              - a plain number -- treated as a precomputed wait in minutes,
                for stations without capacity data yet.
            Stations missing from this dict are assumed to have zero wait.

        soc_step (int, default 10): Granularity, in percentage points, of the
            charge targets the search may choose (only multiples of soc_step
            are ever offered, e.g. 10, 20, 30...) and of the SoC bucketing
            used to merge equivalent states for performance. Smaller = closer
            to truly optimal but slower; larger = faster but coarser charging
            decisions. If soc_step doesn't evenly divide 100, exactly 100%
            will never be offered as a charge target.

        min_destination_soc (float, default 0.0): Minimum SoC (percentage)
            required on arrival at destination. Only constrains the final
            arrival edge -- never forces an intermediate station to charge
            beyond what the remaining route actually needs. Defaults to 0
            (arriving with an empty battery is allowed) for backward
            compatibility; callers will usually want to pass something like
            10-15 in practice.

    Returns:
        On success, a dict:
            {
                'total_time_minutes': float,  # driving + charging + queue waits
                'path': [(node, action, soc_after_action), ...]
                # e.g. ('Station_B', 'charge_to_70%_immediately', 70)
            }
        Returns None if no feasible route exists (e.g. not enough range even
        with charging, or min_destination_soc can't be satisfied).
    """
    # --- Input validation ---
    if not isinstance(stations_graph, dict) or not stations_graph:
        raise ValueError("stations_graph must be a non-empty dict of {node: {neighbor: edge_data}}")

    known_nodes = set(stations_graph.keys())
    for neighbors in stations_graph.values():
        known_nodes.update(neighbors.keys())

    if origin not in known_nodes:
        raise ValueError(
            f"origin {origin!r} does not appear anywhere in stations_graph "
            f"(neither as a node with outgoing edges nor as a neighbor)"
        )
    if destination not in known_nodes:
        raise ValueError(f"destination {destination!r} does not appear anywhere in stations_graph")

    if not (0.0 <= initial_soc <= 100.0):
        raise ValueError(f"initial_soc must be between 0 and 100, got {initial_soc}")

    if not (0.0 <= min_destination_soc <= 100.0):
        raise ValueError(f"min_destination_soc must be between 0 and 100, got {min_destination_soc}")

    if not isinstance(soc_step, int) or soc_step <= 0:
        raise ValueError(f"soc_step must be a positive integer, got {soc_step!r}")

    if not callable(vehicle_profile.get('charge_curve_func')):
        raise ValueError("vehicle_profile must contain a callable 'charge_curve_func'")

    unknown_wait_stations = set(real_time_status.keys()) - known_nodes
    if unknown_wait_stations:
        warnings.warn(
            f"real_time_status references station(s) not found in stations_graph: "
            f"{sorted(unknown_wait_stations)} -- likely a typo; these entries will be ignored"
        )
    # --- end validation ---

    counter = itertools.count()  # heap tie-breaker so paths are never compared
    pq = [(0.0, next(counter), origin, float(initial_soc), [(origin, 'start', float(initial_soc))])]

    # Tracks, per node, the highest SoC among states already expanded there.
    # Because the priority queue pops in non-decreasing order of total_time,
    # any state popped later for the same node is guaranteed to have
    # time >= an earlier pop's time. So it's only worth exploring if its SoC
    # is strictly better than what we've already expanded -- a state with
    # equal-or-worse SoC and equal-or-worse time can never lead anywhere
    # better than the already-expanded state could. This is exact (no
    # discretization/bucketing approximation), and O(1) per check.
    best_soc_seen = {}
    charge_curve = vehicle_profile['charge_curve_func']

    while pq:
        total_time, _, curr_node, curr_soc, path = heapq.heappop(pq)

        if curr_node == destination:
            return {
                'total_time_minutes': total_time,
                'path': path
            }

        if curr_node in best_soc_seen and best_soc_seen[curr_node] >= curr_soc:
            continue
        best_soc_seen[curr_node] = curr_soc

        # --- ACTION 1: CHARGE (Vertical Expansion) ---
        is_station = curr_node != origin and curr_node != destination
        if is_station:
            queue_wait = _estimate_wait(real_time_status.get(curr_node))
            start_target = int((curr_soc // soc_step) + 1) * soc_step
            # 100 is always offered explicitly, even if soc_step doesn't evenly
            # divide 100 and so wouldn't otherwise land on it (e.g. soc_step=30
            # would naturally only reach 30/60/90 without this).
            candidate_targets = sorted(set(range(start_target, 101, soc_step)) | {100})
            for target_soc in candidate_targets:
                if target_soc <= curr_soc:
                    continue

                charge_time = charge_curve(curr_soc, target_soc)
                edge_cost = queue_wait + charge_time

                wait_text = f"after waiting {queue_wait}m" if queue_wait > 0 else "immediately"
                new_path = path + [(curr_node, f'charge_to_{target_soc}%_{wait_text}', target_soc)]
                heapq.heappush(pq, (total_time + edge_cost, next(counter), curr_node, float(target_soc), new_path))

        # --- ACTION 2: DRIVE (Horizontal Expansion) ---
        if curr_node in stations_graph:
            for neighbor, edge_data in stations_graph[curr_node].items():
                soc_cost = edge_data['soc_cost']

                if curr_soc >= soc_cost:
                    next_soc = curr_soc - soc_cost

                    # Safety buffer: don't allow arriving at the destination below threshold.
                    if neighbor == destination and next_soc < min_destination_soc:
                        continue

                    driving_time = edge_data['driving_time']
                    new_path = path + [(neighbor, 'drive_to', round(next_soc, 1))]
                    heapq.heappush(pq, (total_time + driving_time, next(counter), neighbor, next_soc, new_path))

    return None


def standard_ev_charge_curve(from_soc, to_soc):
    """
    Closed-form piecewise-linear charge curve:
    2%/min below 80% SoC, 0.5%/min from 80% to 100%.
    """
    BREAKPOINT = 80.0
    FAST_RATE = 2.0   # % per minute
    SLOW_RATE = 0.5   # % per minute

    if to_soc <= from_soc:
        return 0.0
    if to_soc <= BREAKPOINT:
        return (to_soc - from_soc) / FAST_RATE
    if from_soc >= BREAKPOINT:
        return (to_soc - from_soc) / SLOW_RATE

    fast_portion = (BREAKPOINT - from_soc) / FAST_RATE
    slow_portion = (to_soc - BREAKPOINT) / SLOW_RATE
    return fast_portion + slow_portion


my_ev = {'charge_curve_func': standard_ev_charge_curve}
