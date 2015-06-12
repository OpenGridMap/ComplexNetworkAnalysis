import transform_network as transform
import generate_standard_graphs as mygraphs
import measures_on_simple_data as data
try:
    import matplotlib.pyplot as plt
except:
    raise
import numpy as np
import networkx as nx
import network_utils as utils

def compare(n):
    F = data.analyseNetwork(n);
    dh = nx.degree_histogram(F)
    mat = nx.degree_mixing_matrix(F)
    print(np.array_str(mat))
    coef = nx.degree_assortativity_coefficient(F)
    

    G = transform.generate_degree_distribution(dh)
    G = transform.transform_graph_assortativity_coef(G,mat,coef)
    # G = transform.transform_graph_assortativity(G,mat)
    utils.printStats(G)
    utils.drawNetwork(G,n*100)
    

compare(13)
#compare(34)
#compare(37)
#compare(123)

plt.show()
