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
    # only upper triangular entries
    matrix = np.random.rand(n,n) < p
                    
    #set diagonal and lower triangular entries to zero
    matrix = np.triu(matrix, 1)
            
    return Graph(matrix = matrix)

def create_watts(n = 1, p = 0.1):
    #create list of edges
    #would be an offdiagonal matrix
    init_edge_list = []
    for i in range(n-1):
        init_edge_list.append((i,i+1))
    init_edge_list.append((0,n-1))
    #G = Graph(nnodes = n, edges = init_edge_list)
    #TG = nx.Graph(G.matrix)
    #draw(TG, pos=nx.spring_layout(TG))
    
    new_edge_list = []
    elist = copy.deepcopy(init_edge_list)
    for edge in init_edge_list:
        if random.random() < p:
            new_edge = edge
            #create new edge until a non-existent is found
            while new_edge in elist or new_edge in new_edge_list or new_edge == (edge[0], edge[0])\
            or (new_edge[1], new_edge[0]) in elist or (new_edge[1], new_edge[0]) in new_edge_list:
                new_edge = (edge[0], random.randint(0, n-1))
            elist.remove(edge)
            #set edges (small number, great number)
            if new_edge[0] > new_edge[1]:
                new_edge = (new_edge[1], new_edge[0])
            new_edge_list.append(new_edge)
        else:
            new_edge_list.append(edge)
            
    return  Graph(nnodes = n, edges = new_edge_list)
    
    
    
def create_barabasi(n, n_0):
    matrix = np.zeros((n,n))
    #connect first node to n_0 others
    matrix[0,1:n_0+1] = 1
    #connect second to n_0 others (still not important which)
    matrix[1, 0] = 1
    print matrix
    for new in range(2, n):
        print new
        nodes = range(n)
        nodes.remove(new)
        print nodes
        probs = list(matrix.sum(axis=1))
        probs.pop(new)
        probs = probs/sum(probs)
        print probs
        ind = numpy.random.choice(nodes, size=n_0, replace=False, p=probs)
        matrix[new, ind] = 1
    print matrix
