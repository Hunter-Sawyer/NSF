import heapq
import itertools
import warnings


def plan_ev_route(origin, destination, initial_soc, stations_graph, vehicle_profile,
                   real_time_status, soc_step=10, min_destination_soc=0.0):
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

        real_time_status (dict): {station_name: expected_wait_minutes}. A
            SNAPSHOT taken at planning time, not a time-varying forecast --
            this is intentional (see discussion), and it's why the result is
            optimal for conditions *right now*, not necessarily conditions at
            actual arrival time. Stations missing from this dict are assumed
            to have zero wait.

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
            queue_wait = real_time_status.get(curr_node, 0.0)

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

regional_graph = {
    'Origin': {'Station_A': {'driving_time': 30.0, 'soc_cost': 20.0}},
    'Station_A': {
        'Station_B': {'driving_time': 60.0, 'soc_cost': 50.0},
        'Station_C': {'driving_time': 80.0, 'soc_cost': 60.0}
    },
    'Station_B': {
        'Station_C': {'driving_time': 20.0, 'soc_cost': 15.0},
        'Destination': {'driving_time': 70.0, 'soc_cost': 70.0}
    },
    'Station_C': {'Destination': {'driving_time': 70.0, 'soc_cost': 50.0}},
    'Destination': {}
}


def print_route(result, scenario_name):
    print(f"\n{'=' * 50}\n SCENARIO: {scenario_name}\n{'=' * 50}")
    if not result:
        print("No feasible route found!")
        return
    print(f"Total Trip Time: {round(result['total_time_minutes'], 2)} minutes\n")
    for step in result['path']:
        node, action, soc = step
        print(f"  -> [{node.ljust(12)}] | {action.ljust(35)} | Battery: {round(soc, 1)}%")


if __name__ == "__main__":
    status_empty = {'Station_A': 0.0, 'Station_B': 0.0, 'Station_C': 0.0}
    print_route(plan_ev_route('Origin', 'Destination', 100.0, regional_graph, my_ev, status_empty),
                "Ideal Conditions (regression check, expect 180.0)")

    status_congested = {'Station_A': 0.0, 'Station_B': 45.0, 'Station_C': 0.0}
    print_route(plan_ev_route('Origin', 'Destination', 100.0, regional_graph, my_ev, status_congested),
                "Traffic Event (regression check, expect 195.0)")

    # Bug-fix check: the old forced-20%-floor probe, now with no artificial floor
    tiny_graph = {
        'Origin': {'S': {'driving_time': 10.0, 'soc_cost': 95.0}},
        'S': {'Destination': {'driving_time': 10.0, 'soc_cost': 12.0}},
        'Destination': {}
    }
    print_route(plan_ev_route('Origin', 'Destination', 100.0, tiny_graph, my_ev, {'S': 0.0}),
                "Floor-bug fix check (expect 23.5, was 27.5)")

    # min_destination_soc check: same graph, require arriving with >=15%
    print_route(plan_ev_route('Origin', 'Destination', 100.0, tiny_graph, my_ev, {'S': 0.0},
                               min_destination_soc=15.0),
                "Arrival buffer check (min_destination_soc=15)")

print("\n--- Re-checking the floor-bug claim ---")
# soc_cost to destination is only 8, not 12. Nearest reachable multiple-of-10
# target >= 8 is 10 itself (not 20). This isolates the actual bug from the
# soc_step grid limitation.
tiny_graph_2 = {
    'Origin': {'S': {'driving_time': 10.0, 'soc_cost': 95.0}},  # arrive S with 5%
    'S': {'Destination': {'driving_time': 10.0, 'soc_cost': 8.0}},  # only need 8%
    'Destination': {}
}
print_route(plan_ev_route('Origin', 'Destination', 100.0, tiny_graph_2, my_ev, {'S': 0.0}),
            "v2 (no hardcoded floor): expect stop at 10%, 2.5min charge, 22.5 total")