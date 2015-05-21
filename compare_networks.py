import transform_network as transform
import generate_standard_graphs as mygraphs
import measures_on_simple_data as data
try:
    import matplotlib.pyplot as plt
except:
    raise
import numpy as np
import networkx as nx

F13 = data.analyseNetwork(13);
dh = nx.degree_histogram(F13)

G13 = transform.generate_degree_distribution(dh)

"""PA13 = mygraphs.barabasi_albert_graph_modified(13,1.85)
mygraphs.analyseGraph(PA13, "scalefree_13", 13121)"""
mat = nx.degree_mixing_matrix(G13)
print(np.array_str(mat))
transform.metropolis_dynamics_assortativity(G13,nx.degree_mixing_matrix(F13),nx.degree_assortativity_coefficient(F13))

plt.show()
