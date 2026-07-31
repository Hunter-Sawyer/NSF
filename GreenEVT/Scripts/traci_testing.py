import os
import sys
import random
import sumolib
import traci
import routing # Behavior model

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

def run_traci_simulation(tracker_file=None, route_file="test_PA_rou.xml", net_file="PA.network.net.xml"):
    if tracker_file == None:
        traci_cmd = ["sumo",
                    "--net-file", net_file,
                    "--route-files", route_file,
                    "--additional-files", "./chargers_additional.xml",
                    "--chargingstations-output", "./charging_stations_output.xml",
                    "--time-to-teleport", "-1"]
    else:
        traci_cmd = ["sumo",
                    "--net-file", net_file,
                    "--route-files", route_file,
                    "--additional-files", f"./chargers_additional.xml,{tracker_file}",
                    "--chargingstations-output", "./charging_stations_output.xml",
                    "--time-to-teleport", "-1",
                    "--no-warnings", "true"]
    
    traci.start(traci_cmd)

    # 1. BUILD GRAPH: Parse the target city network dynamically for the routing algorithm

    # NOTE: This builds a naive graph treating every junction as a potential station to scaffold 
    # Dijkstra and state-machine testing. For production, this must be replaced with a meta-network 
    # using sumolib/findRoute to map true paths between strictly valid charging stations and origins/destinations.
    net = sumolib.net.readNet(net_file)
    stations_graph = {}
    for node in net.getNodes():
        node_id = node.getID()
        stations_graph[node_id] = {}
        for edge in node.getOutgoing():
            target_node = edge.getToNode().getID()
            travel_time = edge.getLength() / edge.getSpeed()

            # NOTE: Using a linear proxy (time / 10.0) for energy consumption is a deliberate 
                # simplification to keep debugging deterministic without needing a high-fidelity physical model.
            stations_graph[node_id][target_node] = {
                'driving_time': travel_time / 60.0,
                'soc_cost': travel_time / 10.0, 
                'sumo_edge_id': edge.getID()
            }

    charging_stations = traci.chargingstation.getIDList()
    vehicle_info = {}   

    for step in range(100000):
        traci.simulationStep()
        departed = traci.simulation.getDepartedIDList()
             
        for veh_id in departed:
            if traci.vehicle.getTypeID(veh_id) == "electric_vehicle":
                battery_cap = float(traci.vehicle.getParameter(veh_id, "device.battery.capacity"))
                random_soc = random.gauss(.5 * battery_cap, 0.1 * battery_cap)
                initial_soc = max(0, min(battery_cap, random_soc))
                # initial_soc = 0.05 * battery_cap # Starve the battery to 5% for test
                traci.vehicle.setParameter(veh_id, "device.battery.chargeLevel", str(initial_soc))

                # 2. PROACTIVE ROUTING: Calculate optimal path on departure
                initial_soc_pct = (initial_soc / battery_cap) * 100.0
                route_edges = traci.vehicle.getRoute(veh_id)
                
                # Extract origin and destination nodes from the pre-assigned edges
                start_node = net.getEdge(route_edges[0]).getFromNode().getID()
                end_node = net.getEdge(route_edges[-1]).getToNode().getID()


                # NOTE: Charge demand data (real_time_status) is intentionally bypassed here to prioritize 
                # geographical relocation and ensure baseline stability in the new city layout. 
                # Additionally, the universal vehicle_profile operates purely in percentage space, avoiding 
                # unnecessary variance during core route-generation testing despite differing absolute capacities.
                plan = routing.plan_ev_route(
                    origin=start_node,
                    destination=end_node,
                    initial_soc=initial_soc_pct,
                    stations_graph=stations_graph,
                    vehicle_profile=routing.my_ev,
                    real_time_status={}
                )

                if plan:
                    # 3. QUEUE INITIALIZATION: Store the full sequence of actions
                    vehicle_info[veh_id] = {
                        "route_plan": plan['path'][1:], # Skip the 'start' waypoint
                        "state": "executing_plan",
                        "current_target_soc": None,
                        "current_node": plan['path'][0][0] # Track the current node here
                    }

                    print(f"Vehicle {veh_id} generated plan: {vehicle_info[veh_id]['route_plan']}")

        # 4. DYNAMIC STATE MACHINE
        for veh_id, info in list(vehicle_info.items()):
            
            # If the vehicle needs to pull into a charger
            if info["state"] == "to_charger":
                lane = traci.chargingstation.getLaneID(info["station"])
                if traci.vehicle.getLaneID(veh_id) == lane:
                    info["state"] = "charging"

                    print(f"Vehicle {veh_id} arrived at charger. Target SoC: {info['current_target_soc']}%")


            # 5. DYNAMIC CHARGE TARGET (Replaces the 90% rule)
            elif info["state"] == "charging":
                capacity = float(traci.vehicle.getParameter(veh_id, "device.battery.capacity"))
                currentCharge = float(traci.vehicle.getParameter(veh_id, "device.battery.chargeLevel"))
                current_soc_pct = (currentCharge / capacity) * 100.0
            
                # Unhook exactly when the algorithm's target is reached
                if current_soc_pct >= info["current_target_soc"]:
                    print(f"Vehicle {veh_id} reached target SoC. Resuming route.")
                    info["state"] = "executing_plan"
                    traci.vehicle.resume(veh_id)

            # Advance the queue
            elif info["state"] == "driving":
                # Check if the car has physically reached the target edge
                if traci.vehicle.getRoadID(veh_id) == info["target_edge"]:
                    info["state"] = "executing_plan"

            elif info["state"] == "executing_plan":
                if len(info["route_plan"]) == 0:
                    del vehicle_info[veh_id] # Trip complete
                    continue
                
                next_step = info["route_plan"].pop(0)
                target_node = next_step[0]
                action = next_step[1]
                
                if "charge_to" in action:
                    info["current_target_soc"] = next_step[2]
                    info["state"] = "to_charger"
                    
                    # NOTE: Random station assignment is a temporary fallback to guarantee the state machine 
                    # triggers the `to_charger` and `charging` states. Once the meta-network governs physical nodes, 
                    # the target_node will be the actual station ID, and this randomness must be removed.
                    station_id = random.choice(charging_stations) 
                    info["station"] = station_id
                    
                    lane = traci.chargingstation.getLaneID(station_id)
                    edge_id = traci.lane.getEdgeID(lane)
                    
                    traci.vehicle.changeTarget(veh_id, edge_id)
                    traci.vehicle.setChargingStationStop(veh_id, station_id, duration=10000.0)

                elif "drive_to" in action:
                    edge_id = stations_graph[info["current_node"]][target_node]['sumo_edge_id']
                    traci.vehicle.changeTarget(veh_id, edge_id)
                    
                    info["current_node"] = target_node
                    info["target_edge"] = edge_id   # <--- Track the edge we are driving to
                    info["state"] = "driving"       # <--- Switch state to pause the queue

    traci.close()

if __name__ == "__main__":
    
    run_traci_simulation(route_file = "../genetic_alg/pa_test/intermediate_files/routes/2_34_rou.xml")