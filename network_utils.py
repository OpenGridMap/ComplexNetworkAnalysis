try:
    import matplotlib.pyplot as plt
except:
    raise

import networkx as nx
import numpy as np
from pylab import hist
import pylab as pl
import csv

def degreehisto_to_degreeseq(degree_histogram):
    dd = []
    for x in range(0,len(degree_histogram)):
        dd += [x]*degree_histogram[x]
    return dd

def printComponent(C,path):
    with open(path, 'w+') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', lineterminator='\n')
        writer.writerow(["Node A","Node B","Length(ft.)","Config."])
        length = nx.get_edge_attributes(C, 'length')
        for edge in C.edges():
            print(edge)
            writer.writerow([edge[0],edge[1],length[edge],0])
    return

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
    # nx.draw_networkx_labels(graph,pos,font_size=20,font_family='sans-serif')

    plt.axis('off')

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
    
    plt.draw() # display
    
    return

def analyseGraph(graph, graph_name, id_number):
    print("--------------------------")
    print(graph_name)
    printStatsV(graph)
    drawNetwork(graph, id_number)
    plt.savefig("output/" + graph_name + "_" + str(id_number) + ".png") # save as png
    return graph

def getStats(graph):
    stats = dict()
    stats["Nodes"] = nx.number_of_nodes(graph)
    stats["Edges"] = nx.number_of_edges(graph)
    stats["Neighbors/node"] = 2 * float(stats["Edges"])/ stats["Nodes"]
        
    c = nx.average_clustering(graph)
    stats["Clustering coefficient"] = "%3.2f"%c

    try:
        r = nx.degree_assortativity_coefficient(graph)
        stats["Degree assortativity"] = "%3.2f"%r
        r = get_assortativity_coeff(graph)
        # stats["Degree assortativity - own"] = "%3.2f"%r
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
        p /= i
        stats["Connected components"] = i
        stats["Diameter - sum on cc"] = "%3.2f"%d
        stats["Characteristic path length - avg on cc"] = "%3.2f"%p 
    
    dd = nx.degree_histogram(graph)
    stats["Max degree"] = len(dd) - 1

    return stats

def printStatsV(graph):
    stats = getStats(graph)
    print("----------------------")
    for y in sorted(stats):
        print (stats[y])
    print("----------------------")

    return stats

def printStatsKV(graph):
    stats = getStats(graph)
    print("----------------------")
    for y in sorted(stats):
        print (y,':',stats[y])
    print("----------------------")

    return stats


def get_assortativity_coeff(G):
    a = nx.degree_mixing_matrix(G)
    # a: assortativity matrix
    M = np.matrix(a)
    
    sum_rows = np.add.reduce(M,axis=0)
    sum_lines = np.add.reduce(M,axis=1)
    sum_rows_sq = sum_rows * np.transpose(sum_lines)
    sum_sq = np.add.reduce(sum_rows_sq).item(0)
        
    d = (np.trace(M) - sum_sq) / (1 - sum_sq)
    return d

