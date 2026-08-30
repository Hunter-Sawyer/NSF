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
SIMULATION_DAYS = 5


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


def _route_stage(from_edge, to_edge):
    try:
        stage = traci.simulation.findRoute(from_edge, to_edge)
    except traci.exceptions.TraCIException:
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


def _initialize_battery(
    vehicle_id,
    battery_state,
    home_state,
    current_time,
    home_charge_power_w,
    home_charge_efficiency,
):
    capacity = float(
        traci.vehicle.getParameter(vehicle_id, "device.battery.capacity")
    )
    key = activitygen_vehicle_key(vehicle_id)
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
    return capacity, charge


def _dispatch_vehicle(
    vehicle_id,
    vehicle_info,
    battery_state,
    home_state,
    current_time,
    home_edges,
    home_return_trip_numbers,
    charging_stations,
    reserve_soc,
    estimated_consumption_wh_per_meter,
    charge_target_soc,
    home_charge_power_w,
    home_charge_efficiency,
):
    """Make the charge/no-charge decision for the vehicle's current leg."""
    if traci.vehicle.getTypeID(vehicle_id) != "electric_vehicle":
        return

    route = traci.vehicle.getRoute(vehicle_id)
    if not route:
        return

    household = activitygen_vehicle_key(vehicle_id)
    _remember_home_edge(vehicle_id, route, home_edges)

    capacity, charge = _initialize_battery(
        vehicle_id,
        battery_state,
        home_state,
        current_time,
        home_charge_power_w,
        home_charge_efficiency,
    )
    destination_edge = route[-1]
    choice = choose_least_impact_charger(
        vehicle_id,
        destination_edge,
        charge,
        capacity,
        charging_stations,
        reserve_soc=reserve_soc,
        estimated_consumption_wh_per_meter=estimated_consumption_wh_per_meter,
        charge_target_soc=charge_target_soc,
    )

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

    if choice is None:
        return

    info.update({
        "station": choice["station"],
        "station_edge": choice["edge"],
        "target_charge": choice["target_charge"],
        "state": "to_charger",
    })

    traci.vehicle.changeTarget(vehicle_id, choice["edge"])
    traci.vehicle.setChargingStationStop(
        vehicle_id,
        choice["station"],
        duration=100000,
    )

    print(
        f"{vehicle_id}: charging at {choice['station']} "
        f"(added time={choice['added_time']:.1f}s, "
        f"added distance={choice['added_distance']:.1f}m)"
    )


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
):
    """Run ActivityGen trips and make charging decisions through TraCI.

    ActivityGen remains responsible for home/work routing. TraCI only changes
    the current leg when charging is needed, then restores the ActivityGen
    destination after charging. The ActivityGen schedule is expected to have
    been generated with the same ``simulation_days`` value (five by default).
    """
    if not isinstance(simulation_days, int) or simulation_days <= 0:
        raise ValueError("simulation_days must be a positive integer")

    route_file = _resolve_path(route_file)
    net_file = _resolve_path(net_file)
    additional_files = [SCRIPT_DIR / "chargers_additional.xml"]
    if tracker_file is not None:
        additional_files.append(_resolve_path(tracker_file))

    traci_cmd = [
        "sumo",
        "--net-file", str(net_file),
        "--route-files", str(route_file),
        "--additional-files", ",".join(str(path) for path in additional_files),
        "--chargingstations-output", str(SCRIPT_DIR / "charging_stations_output.xml"),
        "--time-to-teleport", "300",
        "--end", str(simulation_days * 24 * 60 * 60),
        "--no-warnings", "true",
    ]

    traci.start(traci_cmd)
    charging_stations = traci.chargingstation.getIDList()
    vehicle_info = {}
    battery_state = {}
    home_state = {}
    home_edges = {}

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            _remember_battery_levels(battery_state)
            traci.simulationStep()
            current_time = traci.simulation.getTime()

            departed = set(traci.simulation.getDepartedIDList())
            arrived = set(traci.simulation.getArrivedIDList())
            stop_ended = set(
                traci.simulation.getStopEndingVehiclesIDList()
            )

            # SUMO removes a vehicle at the end of its route.  Save the latest
            # SOC so the next trip for that household can receive the energy
            # accumulated while it was at home.
            for vehicle_id in arrived:
                info = vehicle_info.pop(vehicle_id, None)
                if info is None or not info.get("is_home_return"):
                    continue

                charge = battery_state.get(info["activitygen_key"])
                if charge is not None:
                    home_state[info["activitygen_key"]] = {
                        "charge": charge,
                        "arrival_time": current_time,
                    }

            # ActivityGen normally creates a new vehicle ID for each trip.
            # This catches both the home and work legs.
            for vehicle_id in departed:
                route = traci.vehicle.getRoute(vehicle_id)

                # Learn the home edge from every ActivityGen vehicle, not
                # only EVs. EVs are assigned per trip, so an EV may first
                # appear on trip 2 or 3 for a household.
                _remember_home_edge(vehicle_id, route, home_edges)

                _dispatch_vehicle(
                    vehicle_id,
                    vehicle_info,
                    battery_state,
                    home_state,
                    current_time,
                    home_edges,
                    home_return_trip_numbers,
                    charging_stations,
                    reserve_soc,
                    estimated_consumption_wh_per_meter,
                    charge_target_soc,
                    home_charge_power_w,
                    home_charge_efficiency,
                )

            # If ActivityGen represents work as a stop on the same vehicle,
            # the vehicle is leaving through this event rather than a new
            # departed-ID event.
            for vehicle_id in stop_ended:
                info = vehicle_info.get(vehicle_id)
                if info is None or info.pop("ignore_stop_end", False):
                    continue

                _dispatch_vehicle(
                    vehicle_id,
                    vehicle_info,
                    battery_state,
                    home_state,
                    current_time,
                    home_edges,
                    home_return_trip_numbers,
                    charging_stations,
                    reserve_soc,
                    estimated_consumption_wh_per_meter,
                    charge_target_soc,
                    home_charge_power_w,
                    home_charge_efficiency,
                )

            active_vehicle_ids = set(traci.vehicle.getIDList())
            for vehicle_id, info in list(vehicle_info.items()):
                if vehicle_id not in active_vehicle_ids:
                    vehicle_info.pop(vehicle_id)
                    continue

                if info["state"] == "to_charger":
                    station_vehicles = set(
                        traci.chargingstation.getVehicleIDs(info["station"])
                    )
                    if vehicle_id in station_vehicles:
                        info["state"] = "charging"

                elif info["state"] == "charging":
                    charge = float(
                        traci.vehicle.getParameter(
                            vehicle_id,
                            "device.battery.chargeLevel",
                        )
                    )
                    if charge >= info["target_charge"]:
                        traci.vehicle.resume(vehicle_id)
                        traci.vehicle.changeTarget(
                            vehicle_id,
                            info["destination"],
                        )
                        info["ignore_stop_end"] = True
                        info["state"] = "to_destination"

            _remember_battery_levels(battery_state)
    finally:
        traci.close()


if __name__ == "__main__":
    run_traci_simulation(
        route_file="..\\genetic_alg\\pa_test\\intermediate_files\\routes\\2_34_rou.xml",
        tracker_file="..\\genetic_alg\\pa_test\\intermediate_files\\trackers\\2_34_tracker.xml"
    )
