try:
    import matplotlib.pyplot as plt
except:
    raise

import networkx as nx
import network_utils

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


def analyseNetwork(i):
    print("--------------------------")
    print("Feeder" + str(i))
    G = readNetworkData('input/feeder' + str(i) + '/line data.csv')
    network_utils.printStats(G,i)
    network_utils.drawNetwork(G,i)
    plt.savefig("output/network_" + str(i) + ".png") # save as png
    return

analyseNetwork(13)
analyseNetwork(34)
analyseNetwork(37)
analyseNetwork(123)

plt.show()


