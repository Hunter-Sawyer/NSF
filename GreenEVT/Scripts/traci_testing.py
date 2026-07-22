import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci
import random

def run_traci_simulation(tracker_file=None, route_file=".\\test_PA_rou.xml", net_file="..\\data\\Palo_Alto\\PA.network.net.xml"):
    if tracker_file == None:
        traci_cmd = ["sumo",
                    "--net-file", net_file,
                    "--route-files", route_file,
                    "--additional-files", ".\\chargers_additional.xml",
                    "--chargingstations-output", ".\\charging_stations_output.xml",
                    "--time-to-teleport", "-1"]
    if tracker_file != None:
        traci_cmd = ["sumo",
                    "--net-file", net_file,
                    "--route-files", route_file,
                    "--additional-files", f".\\chargers_additional.xml,{tracker_file}",
                    "--chargingstations-output", ".\\charging_stations_output.xml",
                    "--time-to-teleport", "-1",
                    "--no-warnings", "true"]
    
    traci.start(traci_cmd)

    charging_stations = traci.chargingstation.getIDList()
    #print(f"Charging stations: {charging_stations}")

    #charger sanity check
    # for charging_station in charging_stations:
    #     lane = traci.chargingstation.getLaneID(charging_station)
    #     length = traci.lane.getLength(lane)

    #     print(length)
    # exit()
    vehicle_info = {}   

    for step in range(100000):
        #print(f"Simulation step: {step}")
        traci.simulationStep()
        departed = traci.simulation.getDepartedIDList()
        charge_chance = 1
        
             

        for veh_id in departed:
            #print(f"Vehicle {veh_id} has departed.")
            if traci.vehicle.getTypeID(veh_id) == "electric_vehicle":
                battery_cap = float(traci.vehicle.getParameter(veh_id, "device.battery.capacity"))
                random_soc = random.gauss(.5 * battery_cap, 0.1 * battery_cap)
                initial_soc = max(0, min(battery_cap, random_soc))
                traci.vehicle.setParameter(veh_id, "device.battery.chargeLevel", str(initial_soc))

                if initial_soc < 0.2 * battery_cap:
                    route = traci.vehicle.getRoute(veh_id)
                    original_destination = route[-1]

                    #station_id = random.choice(charging_stations)

                    
                    veh_edge = traci.vehicle.getRoadID(veh_id)
                    veh_pos = traci.vehicle.getLanePosition(veh_id)


                    closest_station = None
                    distance_to_closest = float("inf")
                    for station in charging_stations:

                        station_lane = traci.chargingstation.getLaneID(station)
                        station_edge = traci.lane.getEdgeID(station_lane)

                        try:
                            distance = traci.simulation.getDistanceRoad(
                                veh_edge,
                                veh_pos,
                                station_edge,
                                0.0,
                                isDriving=True
                            )
                            if distance >= 0 and distance < distance_to_closest:
                                distance_to_closest = distance
                                closest_station = station
                        except traci.exceptions.TraCIException:
                            pass
                    if closest_station is None:
                        print(f"No reachable charging station found for vehicle {veh_id}.")
                        station_id = random.choice(charging_stations)  # Fallback to a random station if none are reachable
                    else:
                        station_id = closest_station
                

                    vehicle_info[veh_id] = {
                        "destination": original_destination,
                        "station": station_id,
                        "state": "to_charger"
                    }

                    
                    lane = traci.chargingstation.getLaneID(station_id)
                    edge_id = traci.lane.getEdgeID(lane)


                    
                    #print(f"lane: {lane}")
                    #edge = traci.lane.getEdgeID(traci.lane)
                    #print(f"Vehicle {veh_id} is departing and will charge at station {station_id} on lane {lane}.")
                    traci.vehicle.changeTarget(veh_id, edge_id)


                    traci.vehicle.setChargingStationStop(
                        veh_id,
                        station_id,
                        duration = 10000
                    )

        for veh_id, info in vehicle_info.items():
            #print(f"Vehicle {veh_id} is in state {info['state']}, heading to station {info['station']}, location {traci.chargingstation.getLaneID(info['station'])} with current location {traci.vehicle.getLaneID(veh_id)}.")

            if info["state"] == "to_charger":
                # print("____________to_chargers____________")
                lane = traci.chargingstation.getLaneID(info["station"])

                #print(f"vehicle {veh_id} is {traci.vehicle.getLaneID(veh_id)} heading to lane {lane}")
                if traci.vehicle.getLaneID(veh_id) == lane:

                    #print(f"Vehicle {veh_id} has arrived at charging station {info['station']} on lane {lane}.")

                    info["state"] = "charging"

            elif info["state"] == "charging":
                #print(f"Vehicle {veh_id} is currently charging at station {info['station']}.")
                capacity = float(traci.vehicle.getParameter(veh_id, "device.battery.capacity"))
                currentCharge = float(traci.vehicle.getParameter(veh_id, "device.battery.chargeLevel"))
                stateOfCharge = currentCharge / capacity
            
                if stateOfCharge >= 0.9:
                    Original_destination = info["destination"]
                    traci.vehicle.changeTarget(veh_id, Original_destination)
                    info["state"] = "to_destination"
                    #print(f"Vehicle {veh_id} has finished charging and is now heading to its original destination {Original_destination}.")

    traci.close()






if __name__ == "__main__":
    run_traci_simulation(route_file = "..\\genetic_alg\\pa_test\\intermediate_files\\routes\\2_34_rou.xml")