try:
    import matplotlib.pyplot as plt
except:
    raise

import networkx as nx
import numpy as np
from pylab import hist
import pylab as pl

def drawNetwork(graph,k):
    pl.figure(k)
    pl.subplot(211)
    pos = nx.spring_layout(graph) # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(graph,pos,node_size=200)

    # edges
    nx.draw_networkx_edges(graph,pos,
                        width=3)

    # labels
##    nx.draw_networkx_labels(graph,pos,font_size=20,font_family='sans-serif')

    plt.axis('off')
    plt.draw() # display
    return

def printStats(graph,k):
    
    if (nx.is_connected(graph)):
        d = nx.diameter(graph)
        print("Diameter : " + str(d))
        p = nx.average_shortest_path_length(graph)
        print("Characteristic path length : " + "%3.2f"%p)
    else:
        print("Cannot compute diameter and characteristic path length: graph is not connected")

        d = 0.0
        p = 0.0
        i = 0
        for g in nx.connected_component_subgraphs(graph):
            i += 1
            d += nx.diameter(g)
            if len(nx.nodes(g)) > 1:
                p += nx.average_shortest_path_length(g)
        d /= i
        p /= i
        print("There are " + str(i) + " connected component subgraphs in this graph")
        print("Average diameter of connected components : " + "%3.2f"%d)
        print("Average characteristic path length of connected components : " + "%3.2f"%p)
        

    c = nx.average_clustering(graph)
    print("Average clustering : " + "%3.2f"%c)
    
    r = nx.degree_assortativity_coefficient(graph)
    print("Degree assortativity : " + "%3.2f"%r)
    
    # plot degree distribution
    dd = nx.degree_histogram(graph)
    pl.figure(k)
    pl.subplot(212)
    plt.bar(np.arange(len(dd)), dd, width = 0.1)
    plt.axis([0,len(dd),0,max(dd)])
    plt.title("Degree distribution")
    plt.xlabel("degree")
    plt.ylabel("number of nodes")
    return
