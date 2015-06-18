import networkx as nx

def getPath(n):
    return 'input/feeder' + str(n) + '/line data.csv'

def readFeederData(n):
    G = nx.Graph()
    i = 0
    for line in open(getPath(n)):
        i += 1
        if i <= 1:
            continue
        l = line.split(';')
        G.add_edge(l[0],l[1],length=int(l[2]))
        
    return G