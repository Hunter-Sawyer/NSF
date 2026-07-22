import sumolib
import traci
import routing  # Ensure your routing.py file is in the same directory

def run():
    # 1. Parse the generated grid network
    net = sumolib.net.readNet("dummy_grid.net.xml")
    
    # 2. Build the stations_graph dictionary dynamically
    # We will treat SUMO junctions as 'stations' and edges as the roads between them
    stations_graph = {}
    for node in net.getNodes():
        node_id = node.getID()
        stations_graph[node_id] = {}
        
        for edge in node.getOutgoing():
            target_node = edge.getToNode().getID()
            travel_time = edge.getLength() / edge.getSpeed()
            
            # The behavioral model needs driving_time and soc_cost. 
            # We also attach the sumo_edge_id so we know what to tell TraCI later.
            stations_graph[node_id][target_node] = {
                'driving_time': travel_time / 60.0,  # Convert seconds to minutes
                'soc_cost': travel_time / 10.0,      # A dummy battery drain metric for testing
                'sumo_edge_id': edge.getID()
            }

    # 3. Pick random start and end junctions (nodes)
    nodes = list(stations_graph.keys())
    origin = nodes[0]
    destination = nodes[-1]
    
    print(f"Planning EV route from {origin} to {destination}...")

    # 4. Pass the generated graph into your behavioral model
    result = routing.plan_ev_route(
        origin=origin,
        destination=destination,
        initial_soc=100.0,
        stations_graph=stations_graph,
        vehicle_profile=routing.my_ev,
        real_time_status={} # Empty dummy status for queue waits
    )

    if not result:
        print("No route found by the behavioral model.")
        return
        
    print(f"Expected transit time: {result['total_time_minutes']} minutes.")

    # 5. Extract the literal SUMO edges from the behavioral model's path
    sumo_route = []
    current_node = result['path'][0][0]
    
    for step in result['path'][1:]:
        next_node = step[0]
        action = step[1]
        
        # We only need to tell SUMO about the edges the vehicle actually drives across
        if "drive_to" in action:
            edge_id = stations_graph[current_node][next_node]['sumo_edge_id']
            sumo_route.append(edge_id)
            current_node = next_node

    print("Calculated TraCI Edge Route:", sumo_route)

    # 6. Boot up SUMO and inject the vehicle
    # Let sumolib automatically locate sumo-gui using your SUMO_HOME variable
    sumo_binary = sumolib.checkBinary('sumo')
    
    traci.start([sumo_binary, "-c", "dummy.sumocfg"])
    
    traci.route.add("dynamic_ev_route", sumo_route)
    traci.vehicle.add("test_ev", "dynamic_ev_route", typeID="ev_car")
    
    # 7. Step the simulation forward until the vehicle finishes its route
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        
    traci.close()

if __name__ == "__main__":
    run()