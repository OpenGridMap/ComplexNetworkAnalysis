try:
    import matplotlib.pyplot as plt
except:
    raise

import networkx as nx
import network_utils

def generate_scalefree(nb_nodes, nb_edges):
    graph = nx.empty_graph(nb_nodes)
    for nodeA in nx.nodes(graph):
        nodeB = pick_random_node(graph)
        graph.add_edge(nodeA, nodeB)
    return graph

def pick_random_node(graph):
    return nx.nodes(graph)[0]

def generate_smallworld():
    g = nx.empty_graph(nb_nodes)
    return g


def analyseGraph(graph, graph_name, id_number):
    print("--------------------------")
    print(graph_name)
    network_utils.printStats(graph,id_number)
    network_utils.drawNetwork(graph, id_number)
    plt.savefig("output/" + graph_name + "_" + str(id_number) + ".png") # save as png
    return


##randomGraph = nx.gnm_random_graph(13,12)
##analyseGraph(randomGraph, "random_13", 1312)
##
##randomGraph = nx.gnm_random_graph(34,33)
##analyseGraph(randomGraph, "random_34", 3433)
##
##randomGraph = nx.gnm_random_graph(37,36)
##analyseGraph(randomGraph, "random_37", 3736)
##
##randomGraph = nx.gnm_random_graph(123,118)
##analyseGraph(randomGraph, "random_123", 123118)

scaleFreeGraph = generate_scalefree(50,8)
analyseGraph(scaleFreeGraph, "scalefree_5", 58)

plt.show()
