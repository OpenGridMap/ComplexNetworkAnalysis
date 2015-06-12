try:
    import matplotlib.pyplot as plt
except:
    raise

import networkx as nx
import network_utils
import numpy as np

def readNetworkData(path):
    G = nx.Graph()
    nodes = set()

    i = 0
    for line in open(path):
        i += 1
        if i <= 1:
            continue
        l = line.split(';')
        G.add_edge(l[0],l[1],length=int(l[2]))
        
    return G


def analyseNetwork(i):
    print("--------------------------")
    print("Feeder" + str(i))
    G = readNetworkData('input/feeder' + str(i) + '/line data.csv')
    network_utils.printStats(G)
    # mat = nx.degree_mixing_matrix(G)
    # print(np.array_str(mat))
    network_utils.drawNetwork(G,i)
    plt.savefig("output/feeder_" + str(i) + ".png") # save as png
    if not (nx.is_connected(G)):
        k=1
        for g in nx.connected_component_subgraphs(G):
            network_utils.printStats(g)
            network_utils.drawNetwork(g,i+100*k)
            plt.savefig("output/feeder_subcomponent" + str(k) + "_" + str(i) + ".png") # save as png
            k+=1
    
    return G
"""
G13 = analyseNetwork(13)
G34 = analyseNetwork(34)
G37 = analyseNetwork(37)
"""
G123 = analyseNetwork(123)

#plt.show()


