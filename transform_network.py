import random
import numpy as np
from numpy import array_str
import networkx as nx

def generate_degree_distribution(degree_histogram):
    dd = []
    for x in range(0,len(degree_histogram)):
        dd += [x]*degree_histogram[x]
    
    G = nx.random_degree_sequence_graph(dd)
    print("Desired degree distribution : " + str(degree_histogram))
    print("Actual : " + str(nx.degree_histogram(G)))
    return G

def test_assortativity_coeff():
    tab = [[0.258,0.016,0.035,0.013],[0.012,0.157,0.058,0.019],[0.013,0.023,0.306,0.035],[0.005,0.007,0.024,0.016]]
    d = transform.get_assortativity_coeff(tab)
    print(str(d))

def get_assortativity_coeff(a):
    # a: assortativity matrix
    M = np.matrix(a)
    
    sum_rows = np.add.reduce(M,axis=0)
    sum_lines = np.add.reduce(M,axis=1)
    sum_rows_sq = sum_rows * np.transpose(sum_lines)
    sum_sq = np.add.reduce(sum_rows_sq).item(0)
        
    d = (np.trace(M) - sum_sq) / (1 - sum_sq)
    return d

def transform_graph_assortativity_coef(G,A, coeff_d):
# def transform_graph_assortativity(G,A):
    ### G: graph to rewire for desired assortativity coeff (at constant degree distribution)
    ### A: desired assortativity matrix
    
    # coeff_d = get_assortativity_coeff(A)
    
    ### resize matrix of desired assortativity with zeros if actual is bigger
    assortativity_actual = nx.degree_mixing_matrix(G)
    epsilon = len(assortativity_actual) - len(A)
    if epsilon > 0:
        for j in range(len(A)):
            A[j] += [0]*epsilon
            for k in range(epsilon):
                A.append([0]*(len(A)+epsilon))
    # print(array_str(A))
    
    k = 0
    print("Desired assortativity coefficient : " + str(coeff_d))
    # print("Actual before : " + str(get_assortativity_coeff(nx.degree_mixing_matrix(G))))
    print("Actual before : " + str(nx.degree_assortativity_coefficient(G)))
    nb_identical = 0
    prev_delta = 0.2
    delta = 0.2
    rewires = 0
    
    while (abs(delta) > 0.01 and nb_identical < 400): 
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
            rewires += 1
            G.remove_edge(u1,v1)
            G.remove_edge(u2,v2)
            G.add_edge(u1,u2)
            G.add_edge(v1,v2)
            
        # actual_coeff = get_assortativity_coeff(nx.degree_mixing_matrix(G))
        actual_coeff = nx.degree_assortativity_coefficient(G)
        delta = actual_coeff - coeff_d
        if abs(prev_delta - delta) < 0.001:
            nb_identical += 1
        else:
            nb_identical = 0
        prev_delta = delta
        k+= 1
        
    print("Actual after : " + str(actual_coeff))
    print(str(k) + " turn(s), " + str(rewires) + " rewiring(s)")
    
    return G


def transform_graph_assortativity(G,A):
    ### G: graph to rewire for desired assortativity coeff (at constant degree distribution)
    ### A: desired assortativity matrix
    
    coeff_d = get_assortativity_coeff(A)
    
    ### resize matrix of desired assortativity with zeros if actual is bigger
    assortativity_actual = nx.degree_mixing_matrix(G)
    epsilon = len(assortativity_actual) - len(A)
    if epsilon > 0:
        for j in range(len(A)):
            A[j] += [0]*epsilon
            for k in range(epsilon):
                A.append([0]*(len(A)+epsilon))
    # print(array_str(A))
    
    k = 0
    print("Desired assortativity coefficient : " + str(coeff_d))
    print("Actual before : " + str(get_assortativity_coeff(nx.degree_mixing_matrix(G))))
    # print("Actual before : " + str(nx.degree_assortativity_coefficient(G)))
    nb_identical = 0
    prev_delta = 0.2
    delta = 0.2
    rewires = 0
    
    while (abs(delta) > 0.01 and nb_identical < 400): 
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
            rewires += 1
            G.remove_edge(u1,v1)
            G.remove_edge(u2,v2)
            G.add_edge(u1,u2)
            G.add_edge(v1,v2)
            
        actual_coeff = get_assortativity_coeff(nx.degree_mixing_matrix(G))
        # actual_coeff = nx.degree_assortativity_coefficient(G)
        delta = actual_coeff - coeff_d
        if abs(prev_delta - delta) < 0.001:
            nb_identical += 1
        else:
            nb_identical = 0
        prev_delta = delta
        k+= 1
        
    print("Actual after : " + str(actual_coeff))
    print(str(k) + " turn(s), " + str(rewires) + " rewiring(s)")
    
    return G
            
            
    
