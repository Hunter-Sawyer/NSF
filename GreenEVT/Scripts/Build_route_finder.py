from pathlib import Path
import sumolib

def create_route_file(
    network_file = Path("../data/Palo_Alto/PA.network.net.xml"),
    matched_file = Path("matched_edges_0_excluded.txt"),
    output_file = Path("../genetic_alg/PA_test/static_files/Route_finder.xml")):

    net = sumolib.net.readNet(str(network_file))
    valid_edges = {edge.getID() for edge in net.getEdges()}

    edges = []
    for line in matched_file.read_text().splitlines():
        if ":" in line:
            edge_id = line.split(":", 1)[1].strip()
        else:
            edge_id = line.strip()

        if edge_id in valid_edges and edge_id not in edges:
            edges.append(edge_id)

    xml = f'''<additional>
        <edgeData id="Edge_detector"
                file="../outputs/Edge_detector_1.xml"
                edges="{" ".join(edges)}"/>
    </additional>
    '''

    output_file.write_text(xml)
    print(f"Wrote {len(edges)} valid edges to {output_file}")