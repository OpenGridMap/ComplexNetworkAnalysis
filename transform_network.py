import random
try:
    import numpy as np
except ImportError:
    raise ImportError('code requires ',
                          'NumPy: http://scipy.org/')
import networkx as nx
import network_utils as utils
import measures_on_simple_data as data
import infer_parameters as infer
from math import ceil
try:
    import matplotlib.pyplot as plt
except:
    raise

def synthetic(n, connectedDegree = True, keepConnected = True):
    F = data.readFeederData(n);
    dh = nx.degree_histogram(F)
    mat = nx.degree_mixing_matrix(F)
    
    dd = utils.degreehisto_to_degreeseq(dh)
    if connectedDegree:
        G = havel_hakimi_custom_graph(dd)
    else:
        G = nx.random_degree_sequence_graph(dd)
    
    """print("Degree distribution : " + str(dh))
    print("Nb cc : " + str(nx.number_connected_components(G)))
    
    dh2 = nx.degree_histogram(G)
    print("Result Degree distribution : " + str(dh2))"""
    G = transform_graph_assortativity_coef(G,mat,keepConnected)

    return G

def syntheticInferred(n, connectedDegree = True, keepConnected = True):
    dd = infer.infer_degree_sequence(n)
    mat = infer.infer_assortativity_matrix(n)

    if connectedDegree:
        G = havel_hakimi_custom_graph(dd)
    else:
        G = nx.random_degree_sequence_graph(dd)
    """
    print("Degree distribution : " + str(dd))
    print("Nb cc : " + str(nx.number_connected_components(G)))
    
    dh2 = nx.degree_histogram(G)
    print("Result Degree distribution : " + str(dh2))"""
    G = transform_graph_assortativity_coef(G,mat,keepConnected)

    return G

def havel_hakimi_custom_graph(deg_sequence):
    """Return a simple graph with given degree sequence constructed
    using a variant of the Havel-Hakimi algorithm.

    Parameters
    ----------
    deg_sequence: list of integers
        Each integer corresponds to the degree of a node (need not be sorted).

    Raises
    ------
    NetworkXException
        For a non-graphical degree sequence (i.e. one
        not realizable by some simple graph).

    Notes
    -----
    The Havel-Hakimi algorithm constructs a simple graph by
    successively connecting the node of highest degree to other nodes
    of highest degree, resorting remaining nodes by degree, and
    repeating the process. The resulting graph has a high
    degree-associativity.  Here we connect the node of lowest degree to the nodes
    of highest degree in order to get a connected graph.
    Nodes are labeled 1,.., len(deg_sequence),
    corresponding to their position in deg_sequence.

    The basic algorithm is from Hakimi [1]_ and was generalized by
    Kleitman and Wang [2]_.

    References
    ----------
    .. [1] Hakimi S., On Realizability of a Set of Integers as 
       Degrees of the Vertices of a Linear Graph. I,
       Journal of SIAM, 10(3), pp. 496-506 (1962)
    .. [2] Kleitman D.J. and Wang D.L.
       Algorithms for Constructing Graphs and Digraphs with Given Valences
       and Factors  Discrete Mathematics, 6(1), pp. 79-88 (1973) 
    """

    if not (nx.is_valid_degree_sequence(deg_sequence) or nx.is_graphical(deg_sequence) or nx.is_valid_degree_sequence_erdos_gallai(deg_sequence)):
        raise nx.NetworkXError('Invalid degree sequence')

    p = len(deg_sequence)
    G=nx.empty_graph(p)
    num_degs = []
    for i in range(p):
        num_degs.append([])
    dmin, dmax, dsum, n = 10000000, 0, 0, 0
    for d in deg_sequence:
        # Process only the non-zero integers
        if d>0:
            num_degs[d].append(n)
            dmin, dmax, dsum, n = min(dmin,d), max(dmax,d), dsum+d, n+1
    # Return graph if no edges
    if n==0:
        return G

    modstubs = [(0,0)]*(dmax+1)
    # Successively reduce degree sequence by removing the maximum degree
    while n > 0:
        # Retrieve the maximum degree in the sequence
        while len(num_degs[dmax]) == 0:
            dmax -= 1;
        while len(num_degs[dmin]) == 0:
            dmax += 1;
        # If there are not enough stubs to connect to, then the sequence is
        # not graphical
        if dmax > n-1:
            raise nx.NetworkXError('Non-graphical integer sequence')
        
        # Remove most little stub in list
        source = num_degs[dmin].pop()
        n -= 1
        # Reduce the dmin largest stubs
        mslen = 0
        k = dmax
        for i in range(dmin):
            while len(num_degs[k]) == 0:
                k -= 1
            target = num_degs[k].pop()
            G.add_edge(source, target)
            n -= 1
            if k > 1:
                modstubs[mslen] = (k-1,target)
                mslen += 1
        # Add back to the list any nonzero stubs that were removed
        for i in range(mslen):
            (stubval, stubtarget) = modstubs[i]
            num_degs[stubval].append(stubtarget)
            n += 1

    G.name="havel_hakimi_graph %d nodes %d edges"%(G.order(),G.size())
    return G

def test_assortativity_coeff():
    tab = [[0.258,0.016,0.035,0.013],[0.012,0.157,0.058,0.019],[0.013,0.023,0.306,0.035],[0.005,0.007,0.024,0.016]]
    # should be 0.621
    tab = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])/4
    tab = np.matrix([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])/4

    d = numeric_ac(tab)
    print(str(d))

def numeric_ac(mat):
    M = np.matrix(mat)
    # M is a numpy matrix or array
    # numeric assortativity coefficient, pearsonr
    if M.sum() != 1.0:
        M=M/float(M.sum())
    nx,ny=M.shape # nx=ny
    x=np.arange(nx)
    y=np.arange(ny)
    a=np.ravel(M.sum(axis=0))
    b=np.ravel(M.sum(axis=1))
    vara=(a*x**2).sum()-((a*x).sum())**2
    varb=(b*x**2).sum()-((b*x).sum())**2
    xy=np.outer(x,y)
    ab=np.outer(a,b)
    return np.multiply(xy,(M-ab)).sum()/np.sqrt(vara*varb)

def transform_graph_assortativity_coef(G,A,limitateConnComp = True):
    ### G: graph to rewire for desired assortativity coeff (at constant degree distribution)
    ### A: desired assortativity matrix
    coeff_d = numeric_ac(A)
    
    ### resize matrix of desired assortativity with zeros if actual is bigger
    assortativity_actual = nx.degree_mixing_matrix(G)
    epsilon = len(assortativity_actual) - len(A)
    if epsilon > 0:
        for j in range(len(A)):
            A[j] += [0]*epsilon
            for k in range(epsilon):
                A.append([0]*(len(A)+epsilon))    
    k = 0
    print("Desired assortativity coefficient : " + str(coeff_d))
    print("Actual before : " + str(nx.degree_assortativity_coefficient(G)))
    nb_identical = 0
    prev_delta = 0.2
    delta = 0.2
    nb_swaps = 0
    window = 1
    nb_undo_window = 0
    number_cc_prev = nx.number_connected_components(G)
    
    while abs(delta) > 0.01 and nb_identical < 400:
        swapped = []
        countw = 0
        save_nb_identical = nb_identical
        save_delta = delta
        save_prev_delta = prev_delta
        save_nb_swaps = nb_swaps
        while countw < window and delta > 0.01 and nb_identical < 400: 
            nodes = nx.nodes(G)
          
            ### choose two random edges
            u1 = random.choice(nodes)
            neighb = G.neighbors(u1)
            if len(neighb) == 0:
                continue
            v1 = random.choice(neighb)
            j1 = G.degree(u1)
            k1 = G.degree(v1)
            
            u2 = random.choice(nodes)
            neighb = G.neighbors(u2)
            if len(neighb) == 0:
                continue
            v2 = random.choice(neighb)
            j2 = G.degree(u2)
            k2 = G.degree(v2)
            
            if (u1 == u2 or v1 == v2 or u1 == v2 or u2 == v1):
                continue
            if (G.has_edge(u1,u2) or G.has_edge(v1,v2)):
                continue
            
            ### compute rewiring probability
            denom = A[j1][k1]*A[j2][k2]
            p = 1
            if denom != 0:
                p = A[j1][j2]*A[k1][k2]/denom
            
            ### rewire
            if denom == 0 or p > 1 or p >= random.random():
                G.remove_edge(u1,v1)
                G.remove_edge(u2,v2)
                G.add_edge(u1,u2)
                G.add_edge(v1,v2)
                countw += 1
                nb_swaps += 1
                swapped.append((u1,v1,u2,v2))
                
            # actual_coeff = get_assortativity_coeff(nx.degree_mixing_matrix(G))
            actual_coeff = nx.degree_assortativity_coefficient(G)
            delta = abs(actual_coeff - coeff_d)
            if abs(prev_delta - delta) < 0.001:
                nb_identical += 1
            else:
                nb_identical = 0
            prev_delta = delta
            k+= 1

        if limitateConnComp or save_delta < delta:
            nb_cc = nx.number_connected_components(G)
            if nb_cc <= number_cc_prev and save_delta >= delta:
                #window += 1
                number_cc_prev = nb_cc
            else:
                # undo changes from previous window, decrease window
                #print("undo " + str(window) +" " + str(countw))
                while swapped:
                    (u,v,x,y)=swapped.pop()
                    G.add_edge(u,v)
                    G.add_edge(x,y)
                    G.remove_edge(u,x)
                    G.remove_edge(v,y)
                    nb_swaps -= 1
                window = max(int(ceil(float(window)/2)),1)
                
                # restore values from the beginning of the past window
                nb_identical = save_nb_identical
                prev_delta = save_prev_delta
                delta = save_delta
                nb_swaps = save_nb_swaps
                
                nb_undo_window += 1
            if nb_undo_window >= 1000:
                break
        
    print("Actual after : " + str(nx.degree_assortativity_coefficient(G)))
    print(str(k) + " turn(s), " + str(nb_swaps) + " swap(s), " + str(nb_undo_window) + " undo_windows")
    
    return G