try:
    import matplotlib.pyplot as plt
except:
    raise

import networkx as nx
import numpy as np
from pylab import hist
import pylab as pl

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
        nodes.add(l[0])
        nodes.add(l[1])
        
    print("Created graph with " + str(i-1) + " edges and " + str(len(nodes)) + " nodes.")
    return G

def drawNetwork(graph,i):
    pl.figure(i)
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

def printStats(graph,i):
    r = nx.degree_assortativity_coefficient(graph)
    print("Degree assortativity : " + "%3.1f"%r)
    if (nx.is_connected(graph)):
        d = nx.diameter(graph)
        print("Diameter : " + str(d))
        p = nx.average_shortest_path_length(graph)
        print("Characteristic path length : " + "%3.1f"%p)
    else:
        print("Cannot compute diameter and characteristic path length: graph is not connected")
    c = nx.average_clustering(graph)
    print("Average clustering : " + "%3.1f"%c)
    
    # plot degree distribution
    dd = nx.degree_histogram(graph)
    pl.figure(i)
    pl.subplot(212)
    plt.bar(np.arange(len(dd)), dd, width = 0.1)
    plt.axis([0,len(dd),0,max(dd)])
    plt.title("Degree distribution")
    plt.xlabel("degree")
    plt.ylabel("number of nodes")
    return

def analyseNetwork(i):
    print("--------------------------")
    print("Feeder" + str(i))
    G = readNetworkData('feeder' + str(i) + '/line data.csv')
    printStats(G,i)
    drawNetwork(G,i)
    plt.savefig("network_" + str(i) + ".png") # save as png
    return

analyseNetwork(13)
analyseNetwork(34)
analyseNetwork(37)
analyseNetwork(123)

plt.show()


