import networkx as nx

def build_link_graph(links, directed=False):
    """
    Build a graph where each node is a link (edge) from the road network.
    
    Args:
        links: list of dicts or tuples describing road links.
               If tuples: (link_id, u, v)
               If dicts:  {"id":..., "u":..., "v":...}
        directed: if True, connect only if v_i == u_j (successor relation).
                  if False, connect if the two links share any endpoint.
                  
    Returns:
        G_link: networkx Graph (or DiGraph) with nodes=link_ids
    """
    # normalize input
    norm = []
    for item in links:
        if isinstance(item, dict):
            link_id, u, v = item["id"], item["u"], item["v"]
        else:
            link_id, u, v = item
        norm.append((link_id, u, v))
    
    G = nx.DiGraph() if directed else nx.Graph()
    for link_id, u, v in norm:
        G.add_node(link_id, u=u, v=v)

    # index links by endpoints for efficient neighbor finding
    by_u = {}
    by_v = {}
    for link_id, u, v in norm:
        by_u.setdefault(u, []).append(link_id)
        by_v.setdefault(v, []).append(link_id)

    if directed:
        # i -> j if end of i equals start of j
        for link_i, u_i, v_i in norm:
            for link_j in by_u.get(v_i, []):
                if link_i != link_j:
                    G.add_edge(link_i, link_j)
    else:
        # connect if share a node (either start or end)
        # links incident to same node become neighbors
        for node, incident in {**by_u, **by_v}.items():
            # but merging dicts like this loses overlaps; do it properly:
            pass

    if not directed:
        # properly gather incident links per node
        incident_per_node = {}
        for link_id, u, v in norm:
            incident_per_node.setdefault(u, []).append(link_id)
            incident_per_node.setdefault(v, []).append(link_id)
        for node, incident in incident_per_node.items():
            # make clique among incident links
            for i in range(len(incident)):
                for j in range(i + 1, len(incident)):
                    G.add_edge(incident[i], incident[j])

    return G
