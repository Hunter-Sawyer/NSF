import string
import os,sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    
import sumolib
from network_filter import passenger_connectivity

def find_gates(network_path = "../data/sumo_network/greensboro.net.xml"):
    network = sumolib.net.readNet(network_path)
    connectivity = passenger_connectivity(network)
    
    gate_edges = []
    
    for edge in network.getEdges():
        if edge.isSpecial():
            continue

        incoming = edge.getIncoming()
        outgoing = edge.getOutgoing()

        # Keep only one-way boundary edges. Edges with both predecessors and
        # successors are ordinary internal roads and must not receive gate
        # traffic. Passenger-only networks should not use non-drivable edges.
        if not edge.allows("passenger"):
            continue

        # ActivityGen uses an edge with no predecessor as a departure gate and
        # an edge with no successor as an arrival gate.  Check those directed
        # paths explicitly; weak connectivity is insufficient here.
        if not incoming and outgoing and edge.getID() in connectivity["from_hub"]:
            gate_edges.append(edge.getID())
        elif incoming and not outgoing and edge.getID() in connectivity["to_hub"]:
            gate_edges.append(edge.getID())
    return gate_edges
            

if __name__ == "__main__":
    gate_edges = find_gates(r"..\data\Palo_Alto\PA.network.net.xml")
    with open("../genetic_alg/static_files/gates.txt",'w+') as f:
        for item in gate_edges:
            f.write('%s\n' %item)
    f.close()
    
    with open("../genetic_alg/static_files/gates.txt", "r") as f:
        gates = [line.strip() for line in f.readlines() if line.strip()]
        
    
    
    print(len(gates))
