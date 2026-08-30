"""Build a SUMO edgeData additional file from a network and edge list."""

from pathlib import Path
from typing import List, Union
import argparse
import xml.etree.ElementTree as ET

import sumolib


PathLike = Union[str, Path]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_NETWORK_FILE = SCRIPT_DIR.parent / "data" / "Palo_Alto" / "PA.network.net.xml"
DEFAULT_MATCHED_EDGES_FILE = SCRIPT_DIR / "matched_edges_0_excluded.txt"
DEFAULT_OUTPUT_FILE = (
    SCRIPT_DIR.parent / "genetic_alg" / "PA_test" / "static_files" / "Route_finder.xml"
)


def rebuild_route_finder(
    network_file: PathLike = DEFAULT_NETWORK_FILE,
    matched_edges_file: PathLike = DEFAULT_MATCHED_EDGES_FILE,
    output_file: PathLike = DEFAULT_OUTPUT_FILE,
    detector_id: str = "Edge_detector",
    detector_output: str = "../outputs/Edge_detector_1.xml",
) -> List[str]:
    """Create a Route_finder.xml file containing only valid network edges.

    ``matched_edges_file`` may contain either one edge ID per line or
    ``measurement:edge_id`` lines, such as the project's matched-edge files.

    Returns the valid, unique edge IDs in their original order. Edges present
    in the matched-edge file but absent from the network are skipped.
    """
    network_path = Path(network_file)
    matched_path = Path(matched_edges_file)
    output_path = Path(output_file)

    net = sumolib.net.readNet(str(network_path))
    valid_network_edges = {edge.getID() for edge in net.getEdges()}

    valid_edges: List[str] = []
    seen = set()

    for raw_line in matched_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        edge_id = line.split(":", 1)[1].strip() if ":" in line else line
        if edge_id in valid_network_edges and edge_id not in seen:
            valid_edges.append(edge_id)
            seen.add(edge_id)

    root = ET.Element("additional")
    edge_data = ET.SubElement(root, "edgeData")
    edge_data.set("id", detector_id)
    edge_data.set("file", detector_output)
    edge_data.set("edges", " ".join(valid_edges))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(
        output_path,
        encoding="utf-8",
        xml_declaration=True,
    )

    return valid_edges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network",
        default=DEFAULT_NETWORK_FILE,
        help=f"Current SUMO .net.xml file (default: {DEFAULT_NETWORK_FILE})",
    )
    parser.add_argument(
        "--matched",
        default=DEFAULT_MATCHED_EDGES_FILE,
        help=f"Matched edge list (default: {DEFAULT_MATCHED_EDGES_FILE})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Route_finder.xml output path (default: {DEFAULT_OUTPUT_FILE})",
    )
    args = parser.parse_args()

    edges = rebuild_route_finder(args.network, args.matched, args.output)
    print(f"Wrote {len(edges)} valid edges to {args.output}")


if __name__ == "__main__":
    main()
