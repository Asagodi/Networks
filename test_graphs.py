import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import graphs
import types
%matplotlib inline  

#G = create_erdos(0, 0.5)
#TG = nx.Graph(G.matrix)
#draw(TG, pos=nx.spring_layout(TG))

#G = create_watts(1, 0.5)
#TG = nx.Graph(G.matrix)
#draw(TG, pos=nx.spring_layout(TG))

G = create_barabasi(10, 2)
TG = nx.Graph(G.matrix)
draw(TG, pos=nx.spring_layout(TG))

G.matrix_to_nodedict()
G.find_path(0,10)
G.find_all_paths(5,4)
G.find_shortest_path(3,7)

