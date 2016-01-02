import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
%matplotlib inline  

###Check Erdos-Renyi Graph
G = create_erdos(8, 0.5)
#TG = nx.Graph(G.matrix)
#draw(TG, pos=nx.spring_layout(TG))

#G.find_path(0,4)
#G.find_all_paths(5,4)
#G.find_shortest_path(3,7)
#print G.diameter()


###Check Watts Graph
#G = create_watts(10, 0.5)
#TG = nx.Graph(G.matrix)
#draw(TG, pos=nx.spring_layout(TG))

#G.find_path(0,9)
#G.find_all_paths(5,4)
#G.find_shortest_path(3,7)
#print G.diameter()

### Check Barabasi Graph
#G = create_barabasi(20, 1)
#TG = nx.Graph(G.matrix)
#draw(TG, pos=nx.spring_layout(TG))

#G.matrix_to_nodedict()
#G.find_path(0,10)
#G.find_all_paths(5,4)
#G.find_shortest_path(3,7)
#print G.diameter()



