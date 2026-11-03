"""Helpers for keeping ActivityGen inputs inside one connected SUMO network."""


def largest_passenger_component(network):
    """Return edge IDs in the largest weakly connected passenger component."""
    edges = [
        edge
        for edge in network.getEdges()
        if not edge.isSpecial() and edge.allows("passenger")
    ]

    node_edges = {}
    for edge in edges:
        from_id = edge.getFromNode().getID()
        to_id = edge.getToNode().getID()
        node_edges.setdefault(from_id, []).append(edge)
        node_edges.setdefault(to_id, []).append(edge)

    component_edges = set()
    visited_nodes = set()

    for start_node in node_edges:
        if start_node in visited_nodes:
            continue

        stack = [start_node]
        visited_nodes.add(start_node)
        current_edges = set()

        while stack:
            node_id = stack.pop()
            for edge in node_edges[node_id]:
                current_edges.add(edge.getID())
                for adjacent in (edge.getFromNode().getID(), edge.getToNode().getID()):
                    if adjacent not in visited_nodes:
                        visited_nodes.add(adjacent)
                        stack.append(adjacent)

        if len(current_edges) > len(component_edges):
            component_edges = current_edges

    return component_edges


def _reachable_edges(start_edge, forward):
    """Return passenger edges reachable from ``start_edge`` in one direction."""
    reachable = set()
    stack = [start_edge]

    while stack:
        edge = stack.pop()
        edge_id = edge.getID()
        if edge_id in reachable:
            continue
        if edge.isSpecial() or not edge.allows("passenger"):
            continue

        reachable.add(edge_id)
        neighbors = edge.getOutgoing() if forward else edge.getIncoming()
        stack.extend(neighbor for neighbor in neighbors if neighbor.getID() not in reachable)

    return reachable


def passenger_connectivity(network):
    """Return weak, directed, and round-trip passenger connectivity sets.

    ActivityGen chooses arbitrary origin/destination pairs from the supplied
    street list.  The weak component is not enough for a one-way network: two
    edges can be connected through their junctions while no directed route
    exists between them.  A hub-based directed check gives us safe sets for
    population streets and city gates.
    """
    weak_edges = largest_passenger_component(network)
    candidates = [
        edge
        for edge in network.getEdges()
        if edge.getID() in weak_edges
        and not edge.isSpecial()
        and edge.allows("passenger")
        and edge.getIncoming()
        and edge.getOutgoing()
    ]

    if not candidates:
        return {
            "weak": weak_edges,
            "from_hub": set(),
            "to_hub": set(),
            "round_trip": set(),
            "hub": None,
        }

    hub = max(
        candidates,
        key=lambda edge: (
            len(edge.getIncoming()) + len(edge.getOutgoing()),
            edge.getLength(),
        ),
    )
    from_hub = _reachable_edges(hub, forward=True)
    to_hub = _reachable_edges(hub, forward=False)

    return {
        "weak": weak_edges,
        "from_hub": from_hub,
        "to_hub": to_hub,
        "round_trip": from_hub & to_hub,
        "hub": hub.getID(),
    }
