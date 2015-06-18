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

utils.printStatsV(msd.readFeederData(124))
utils.printStatsV(msd.readFeederData(125))
utils.printStatsV(msd.readFeederData(126))
utils.printStatsV(msd.readFeederData(127))
utils.printStatsV(msd.readFeederData(128))
"""


utils.printStatsV(transform.synthetic(13))
utils.printStatsV(transform.synthetic(34))
utils.printStatsV(transform.synthetic(37))
utils.printStatsV(transform.synthetic(123))

### subcomponents of feeder 123
"""
utils.printStatsV(transform.synthetic(124, True))
utils.printStatsV(transform.synthetic(125, True))
utils.printStatsV(transform.synthetic(126, True))
utils.printStatsV(transform.synthetic(127, True))
utils.printStatsV(transform.synthetic(128, True))
"""

#plt.show()

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