import network_utils
import generate_standard_graphs as mygraphs
import time
import networkx as nx
try:
    import matplotlib.pyplot as plt
except:
    raise


def analyseGraph(graph, graph_name, id_number):
    print("--------------------------")
    print(graph_name)
    network_utils.printStats(graph,id_number)
    network_utils.drawNetwork(graph, id_number)
    plt.savefig("output/" + graph_name + "_" + str(id_number) + ".png") # save as png
    return


randomGraph = nx.gnm_random_graph(13,12)
analyseGraph(randomGraph, "random_13", 1312)

randomGraph = nx.gnm_random_graph(34,33)
analyseGraph(randomGraph, "random_34", 3433)

randomGraph = nx.gnm_random_graph(37,36)
analyseGraph(randomGraph, "random_37", 3736)

randomGraph = nx.gnm_random_graph(123,118)
analyseGraph(randomGraph, "random_123", 123118)

scaleFreeGraph = mygraphs.barabasi_albert_graph_modified(13,1.08)
analyseGraph(scaleFreeGraph, "scalefree_13", 1312)

scaleFreeGraph = mygraphs.barabasi_albert_graph_modified(34,1.03)
analyseGraph(scaleFreeGraph, "scalefree_34", 3433)

scaleFreeGraph = mygraphs.barabasi_albert_graph_modified(37,1.03)
analyseGraph(scaleFreeGraph, "scalefree_37", 3736)

scaleFreeGraph = mygraphs.barabasi_albert_graph_modified(123,1.04)
analyseGraph(scaleFreeGraph, "scalefree_123", 123118)

smallWorldGraph = mygraphs.watts_strogatz_graph_modified(13,1.08,0.3)
analyseGraph(smallWorldGraph, "smallWorld_13",1312)

smallWorldGraph = mygraphs.watts_strogatz_graph_modified(34,1.03,0.3)
analyseGraph(smallWorldGraph, "smallWorld_34",3433)

smallWorldGraph = mygraphs.watts_strogatz_graph_modified(37,1.03,0.3)
analyseGraph(smallWorldGraph, "smallWorld_37",3736)

smallWorldGraph = mygraphs.watts_strogatz_graph_modified(123,1.04,0.3)
analyseGraph(smallWorldGraph, "smallWorld_123",123118)

plt.show()
