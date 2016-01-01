import numpy
import math
import random
import copy


class Node:
    """ 
    Useful for the opinion network.
    """
    def __init__(self, value = 0):
        value = value



#Create three possibilities to represent graph?
class Graph:
    def __init__(self, matrix = None, nnodes = 1, nodes = None, edges = None):
        if matrix != None:
            self.matrix = matrix
            self.nnodes = matrix.shape[0]
            #self.nodes = 
            #self.edges = 
        elif nodes == None:
            self.nodes = []
            self.edges = edges
            self.nnodes = nnodes
            self.matrix = np.zeros((nnodes,nnodes))
            self.edgelist_to_matrix()
            
    def edgelist_to_matrix(self):
        #Convert the list of tuples representing edges to a matrix representation of the graph.
        self.matrix = np.zeros((self.nnodes,self.nnodes))
        for edge in self.edges:
            self.matrix[edge[0], edge[1]] = 1
        
  
  
  
#Creating different graphs
def create_erdos(n = 1, p = 1.):
    # n Number of Nodes, p Choice to get edge
    matrix = np.random.rand(n,n) < p
                    
    #set diagonal to zero
    for d in range(n):
        matrix[d, d] = 0
            
    return Graph(matrix = matrix)

def create_watts(n = 1, p = 0.1):
    #create list of edges
    #would be an offdiagonal matrix
    init_edge_list = []
    for i in range(n-1):
        init_edge_list.append((i,i+1))
    for i in range(1, n):
        init_edge_list.append((i,i-1))
    
    new_edge_list = []
    elist = copy.deepcopy(init_edge_list)
    for edge in init_edge_list:
        if random.random() < p:
            new_edge = edge
            while new_edge in elist or new_edge in new_edge_list:
                new_edge = (edge[0], random.randint(0, n-1))
            elist.remove(edge)
            new_edge_list.append(new_edge)
        else:
            new_edge_list.append(edge)
            
    return  Graph(nnodes = n, edges = new_edge_list)
    
