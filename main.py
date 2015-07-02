import measures_on_simple_data as msd
import transform_network as transform
import network_utils as utils
from networkx import number_connected_components, connected_component_subgraphs
from math import sqrt
try:
    import matplotlib.pyplot as plt
except:
    raise


  
"""
G13 = msd.readFeederData(13)
G34 = msd.readFeederData(34)
G37 = msd.readFeederData(37)
G123 = msd.readFeederData(123)
"""

# print stats, draw and save figures
def drawAndSave():
    utils.analyseGraph(msd.readFeederData(13), "feeder13", 13)
    utils.analyseGraph(msd.readFeederData(34), "feeder34", 34)
    utils.analyseGraph(msd.readFeederData(37), "feeder37", 37)
    utils.analyseGraph(msd.readFeederData(123), "feeder123", 123)
    
    utils.analyseGraph(transform.synthetic(13), "synthetic13", 130)
    utils.analyseGraph(transform.synthetic(34), "synthetic34", 340)
    utils.analyseGraph(transform.synthetic(37), "synthetic37", 350)
    utils.analyseGraph(transform.synthetic(123), "synthetic123", 1230)
    
    ### subcomponents of feeder 123 
    utils.analyseGraph(msd.readFeederData(124), "feeder124", 124)
    utils.analyseGraph(msd.readFeederData(125), "feeder125", 125)
    utils.analyseGraph(msd.readFeederData(126), "feeder126", 126)
    utils.analyseGraph(msd.readFeederData(127), "feeder127", 127)
    utils.analyseGraph(msd.readFeederData(128), "feeder128", 128)
    
    utils.analyseGraph(transform.synthetic(124, True), "synthetic_sub124", 12400)
    utils.analyseGraph(transform.synthetic(125, True), "synthetic_sub125", 12500)
    utils.analyseGraph(transform.synthetic(126, True), "synthetic_sub126", 12600)
    utils.analyseGraph(transform.synthetic(127, True), "synthetic_sub127", 12700)
    utils.analyseGraph(transform.synthetic(128, True), "synthetic_sub128", 12800)

def inferParameters(n):    
    utils.analyseGraph(transform.syntheticInferred(n, True), "synthetic_" + str(n), n * 1000)

# print stats of all real feeders and synthetic networks
def printStats():
    utils.printStatsV(transform.synthetic(13))
    utils.printStatsV(transform.synthetic(34))
    utils.printStatsV(transform.synthetic(37))
    utils.printStatsV(transform.synthetic(123))
    
    ### subcomponents of feeder 123
    utils.printStatsV(msd.readFeederData(124))
    utils.printStatsV(msd.readFeederData(125))
    utils.printStatsV(msd.readFeederData(126))
    utils.printStatsV(msd.readFeederData(127))
    utils.printStatsV(msd.readFeederData(128))
    
    utils.printStatsV(transform.synthetic(124, True))
    utils.printStatsV(transform.synthetic(125, True))
    utils.printStatsV(transform.synthetic(126, True))
    utils.printStatsV(transform.synthetic(127, True))
    utils.printStatsV(transform.synthetic(128, True))

def teststat(k):
    nb_cc = []
    for i in range(0,30):
        nb_cc.append(number_connected_components(transform.synthetic(k, True)))
        #nb_cc.append(i)
    print(nb_cc)
    mean = sum(nb_cc)/len(nb_cc)
    print(mean)
    dev = [(x - mean)*(x-mean) for x in nb_cc]
    std_dev = sqrt(sum(dev)/len(nb_cc))
    print(std_dev)
    return

# teststat(124)
inferParameters(32)

#drawAndSave()
#plt.show()