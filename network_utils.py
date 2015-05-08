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
    #pos = nx.shell_layout(graph) # positions for all nodes
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
    nn = nx.number_of_edges(graph)*2.0/nx.number_of_nodes(graph)

    stats = dict()
    stats["Nodes"] = nx.number_of_nodes(graph)
    stats["Edges"] = nx.number_of_edges(graph)
    stats["Neighbors/node"] = "%3.2f"%nn
        
    c = nx.average_clustering(graph)
    stats["Clustering coefficient"] = "%3.2f"%c

    try:
        r = nx.degree_assortativity_coefficient(graph)
        stats["Degree assortativity"] = "%3.2f"%r
    except:
        print("Impossible to compute degree assortativity")
    
    if (nx.is_connected(graph)):
        stats['Diameter'] = nx.diameter(graph)
        p = nx.average_shortest_path_length(graph)
        stats["Characteristic path length"] = "%3.2f"%p
        stats["Connected components"] = 1
    else:
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
        stats["Connected components"] = i
        stats["Diameter - avg on connected components"] = "%3.2f"%d
        stats["Characteristic path length - avg on connected components"] = "%3.2f"%p 


    for y in sorted(stats):
        print (y,':',stats[y])
        #print (stats[y])

    
    # plot degree distribution
    dd = nx.degree_histogram(graph)
    fig = pl.figure(k)
    ax = pl.subplot(212)
    plt.bar(np.arange(len(dd)), dd, width = 0.1)
    plt.axis([0,len(dd),0,max(dd)])
    plt.title("Degree distribution")
    plt.xlabel("degree")
    plt.ylabel("number of nodes")
    #plt.figtext(2, 6, stats, fontsize=15)
    return stats
