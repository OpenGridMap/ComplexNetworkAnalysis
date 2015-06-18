import networkx as nx
import random
import network_utils
try:
    import matplotlib.pyplot as plt
except:
    raise

def analyseGraph(graph, graph_name, id_number):
    print("--------------------------")
    print(graph_name)
    network_utils.printStatsV(graph)
    network_utils.drawNetwork(graph, id_number)
    plt.savefig("output/" + graph_name + "_" + str(id_number) + ".png") # save as png
    return graph

def my_barabasi_albert_graph(n, m, seed=None):
    """ m : int """
    return nx.barabasi_albert_graph(n, m, seed)

def barabasi_albert_graph_modified(n, m, seed=None):
    """Return random graph using Barabási-Albert preferential attachment model.

    A graph of n nodes is grown by attaching new nodes each with m
    edges that are preferentially attached to existing nodes with high
    degree.

    Parameters
    ----------
    n : int
        Number of nodes
    m : double
        Number of edges to attach from a new node to existing nodes
    seed : int, optional
        Seed for random number generator (default=None).

    Returns
    -------
    G : Graph

    Notes
    -----
    The initialization is a graph with m nodes and no edges.

    References
    ----------
    .. [1] A. L. Barabási and R. Albert "Emergence of scaling in
       random networks", Science 286, pp 509-512, 1999.
    """

    residual = m/2 - int(m/2)

    if m < 1 or  m >=n-1:
        raise nx.NetworkXError(\
              "Barabási-Albert network must have m>=1 and m<n, m=%d,n=%d"%(m,n))
    if seed is not None:
        random.seed(seed)

    m_tmp = int(m/2)
    if random.random() <= residual:
        m_tmp += 1
        
    # Add m initial nodes (m0 in barabasi-speak)
    G=nx.empty_graph(m_tmp)
    G.name="barabasi_albert_graph(%s,%s)"%(n,m)
    # Target nodes for new edges
    if m_tmp < 1:
        m_tmp = 1
    targets = list(range(m_tmp))

    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes=[]
    # Start adding the other n-m nodes. The first node is m.
    source=m_tmp
    while source<n:
        source += 1
        G.add_node(source)
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source]*m_tmp,targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source]*m_tmp)

        m_tmp = int(m/2)
        if random.random() <= residual:
            m_tmp += 1
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachement)
        if m_tmp > 0:
            if len(repeated_nodes) == 0 :
                targets = list(range(m_tmp))
            else: 
                targets = _random_subset(repeated_nodes, m_tmp)
        else:
            targets = []
    return G

def _random_subset(seq,u):
    """ Return u unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.
    """
    targets=set()
    while len(targets)<u:
        x=random.choice(seq)
        targets.add(x)
    return targets

def watts_strogatz_graph_modified(n, k, p, seed=None):
    """Return a Watts-Strogatz small-world graph.

    Parameters
    ----------
    n : int
        The number of nodes
    k : int
        Each node is connected to k nearest neighbors in ring topology
    p : float
        The probability of rewiring each edge
    seed : int, optional
        Seed for random number generator (default=None)

    Notes
    -----
    First create a ring over n nodes.  Then each node in the ring is
    connected with its int(k) nearest neighbors (int(k-1) neighbors if k is odd).
    Then connect each node with one more neighbor with probability k/2-int(k/2).
    Then shortcuts are created by replacing some edges as follows:
    for each edge u-v in the underlying "n-ring with k nearest neighbors"
    with probability p replace it with a new edge u-w with uniformly
    random choice of existing node w.

    In contrast with newman_watts_strogatz_graph(), the random
    rewiring does not increase the number of edges. The rewired graph
    is not guaranteed to be connected as in connected_watts_strogatz_graph().

    References
    ----------
    .. [1] Duncan J. Watts and Steven H. Strogatz,
       Collective dynamics of small-world networks,
       Nature, 393, pp. 440--442, 1998.
    """
    if k>=n:
        raise nx.NetworkXError("k>=n, choose smaller k or larger n")
    if seed is not None:
        random.seed(seed)

    residual = k/2 - int(k/2)

    G = nx.Graph()
    G.name="watts_strogatz_graph(%s,%s,%s)"%(n,k,p)
    nodes = list(range(n)) # nodes are labeled 0 to n-1
    G.add_nodes_from(nodes)

    # connect each node to k/2 neighbors
    for j in range(1, int(k)//2 + 1):
        targets = nodes[j:] + nodes[0:j] # first j nodes are now last in list
        G.add_edges_from(zip(nodes,targets))

    for j in range(len(nodes)):
        if random.random() < residual:
            G.add_edge(nodes[j], nodes[(j + int(k)//2 + 1)%len(nodes)])

    # rewire edges from each node
    # loop over all nodes in order (label) and neighbors in order (distance)
    # no self loops or multiple edges allowed
    for e in G.edges():
        if random.random() < p:
            u = e[0]
            v = e[1]
            w = random.choice(nodes)
            # Enforce no self-loops or multiple edges
            while w == u or G.has_edge(u, w):
                w = random.choice(nodes)
                if G.degree(u) >= n-1:
                    break # skip this rewiring
            else:
                G.remove_edge(u,v)
                G.add_edge(u,w)

    return G

def random_graph(nodes, neighbours_per_node):
    return nx.gnm_random_graph(nodes, int(neighbours_per_node * nodes / 2))
