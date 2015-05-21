import random
from numpy import array_str
import networkx as nx

def generate_degree_distribution(degree_histogram):
    
    dd = []
    for x in range(0,len(degree_histogram)-1):
        dd += [x]*degree_histogram[x]
    print(dd)
    
    G = nx.random_degree_sequence_graph(dd)
    print(nx.degree_histogram(G))
    return G

def metropolis_dynamics_assortativity(G,A,d):
    # G: graph to rewire for desired assortativity coeff (at constant degree distribution)
    # A: desired assortativity matrix
    # d: desired assortativity coeff
    
    # resize matrix of desired assortativity with zeros if actual is bigger
    size_A = len(A) - 1
    assortativity_actual = nx.degree_mixing_matrix(G)
    size_actual = len(assortativity_actual) - 1
    epsilon = size_actual - size_A
    print(str(epsilon))
    if epsilon > 0:
        for j in range(len(A)):
            for k in range(epsilon):
                A[j] += [0]
                A.append([0]*(len(A)+epsilon))
    print(array_str(A))
    
    k = 0
    print("Desired assortativity coefficient: " + str(d))
    nb_identical = 0
    prev = 0.2
    delta = 0.2
    rewires = 0
    
    while (abs(delta) > 0.1 and nb_identical < 400): 
        nodes = nx.nodes(G)
      
        # choose two random edges
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
        
        # compute rewiring proba
        denom = A[j1][k1]*A[j2][k2]
        p = A[j1][j2]*A[k1][k2]/denom
        
        # rewire
        if denom == 0 or p > 1 or p >= random.random():
            rewires += 1
            G.remove_edge(u1,v1)
            G.remove_edge(u2,v2)
            G.add_edge(u1,u2)
            G.add_edge(v1,v2)
            
        a = nx.degree_assortativity_coefficient(G)
        delta = a - d
        if abs(prev - delta) < 0.001:
            nb_identical += 1
        else:
            nb_identical = 0
        prev = delta
        k+= 1
        
    print("Actual: " + str(a))
    print(str(k) + " turn(s), " + str(rewires) + " rewiring(s)")
    return G
            
            
    
