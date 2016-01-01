import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
%matplotlib inline  

G = create_erdos(10, 0.2)
print G.matrix

TG = nx.DiGraph(G.matrix)

draw(TG, pos=nx.spring_layout(TG))
