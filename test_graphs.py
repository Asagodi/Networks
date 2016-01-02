import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
%matplotlib inline  

#G = create_erdos(5, 0.5)
#print G.matrix
#TG = nx.Graph(G.matrix)
#draw(TG, pos=nx.spring_layout(TG))

#G = create_watts(10, 0.5)
#print G.matrix
#TG = nx.Graph(G.matrix)
#draw(TG, pos=nx.spring_layout(TG))

G = create_barabasi(10, 1)
TG = nx.Graph(G.matrix)
draw(TG, pos=nx.spring_layout(TG))

