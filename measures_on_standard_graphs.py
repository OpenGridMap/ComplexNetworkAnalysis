import generate_standard_graphs as mygraphs
import time
import networkx as nx
try:
    import matplotlib.pyplot as plt
except:
    raise

randomGraph = nx.gnm_random_graph(13,12)
mygraphs.analyseGraph(randomGraph, "random_13", 13120)

randomGraph = nx.gnm_random_graph(34,33)
mygraphs.analyseGraph(randomGraph, "random_34", 34330)

randomGraph = nx.gnm_random_graph(37,36)
mygraphs.analyseGraph(randomGraph, "random_37", 37360)

randomGraph = nx.gnm_random_graph(123,118)
mygraphs.analyseGraph(randomGraph, "random_123", 1231180)

scaleFreeGraph = mygraphs.barabasi_albert_graph_modified(13,1.85)
mygraphs.analyseGraph(scaleFreeGraph, "scalefree_13", 13121)

scaleFreeGraph = mygraphs.barabasi_albert_graph_modified(34,1.94)
mygraphs.analyseGraph(scaleFreeGraph, "scalefree_34", 34331)

scaleFreeGraph = mygraphs.barabasi_albert_graph_modified(37,1.95)
mygraphs.analyseGraph(scaleFreeGraph, "scalefree_37", 37361)

scaleFreeGraph = mygraphs.barabasi_albert_graph_modified(123,1.92)
mygraphs.analyseGraph(scaleFreeGraph, "scalefree_123", 1231181)

smallWorldGraph = mygraphs.watts_strogatz_graph_modified(13,1.85,0.08)
mygraphs.analyseGraph(smallWorldGraph, "smallWorld_13",13122)

smallWorldGraph = mygraphs.watts_strogatz_graph_modified(34,1.94,0.08)
mygraphs.analyseGraph(smallWorldGraph, "smallWorld_34",34332)

smallWorldGraph = mygraphs.watts_strogatz_graph_modified(37,1.95,0.08)
mygraphs.analyseGraph(smallWorldGraph, "smallWorld_37",37362)

smallWorldGraph = mygraphs.watts_strogatz_graph_modified(123,1.92,0.08)
mygraphs.analyseGraph(smallWorldGraph, "smallWorld_123",1231182)

plt.show()
