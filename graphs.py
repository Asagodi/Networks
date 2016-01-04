import numpy as np
import math
import random
import copy

#Create three possibilities to represent graph?
#TODO: create possibility to give nodes names? or just number-name dict?
#edgelist with weighted graph possibility
class Graph(object):
    """
    Input can be as a matrix, list of edges as tuples,
    dictionary of connections between nodes.
    Matrix should be upper triangular for both directed
    and undirected graphs. Edges without duplicates 
    (also for directed graph). Nondict with all connections.
    """
    def __init__(self, matrix = [], nnodes = 1, nodes = [], \
                 edges = [], nodevalues = [], edgevalues = [],\
                 nodedict = {}, directed = False, weightdict = {}):
        if matrix != []:
            assert np.allclose(matrix, np.triu(matrix)), "Matrix is not triangular"
            self.matrix = matrix
            if not directed:
                self.matrix = self.matrix + self.matrix.T - np.diag(self.matrix.diagonal())
            self.nnodes = matrix.shape[0]
            self.matrix_to_edgelist()
            self.nodedict = self.matrix_to_nodedict()
            self.nodes = range(self.matrix.shape[0])
#             self.weightdict = 
        elif nodes == []:
            #check duplicates if undirected?
            self.edges = edges
            if not directed:
                backedges = [(end, start) for (start, end) in edges]
                self.edges.extend(backedges)
            self.nnodes = nnodes
            if edgevalues == []:
                edgevalues = [1] * len(self.edges)
            self.edgedict = dict(zip(self.edges, edgevalues))
            self.matrix = np.zeros((nnodes,nnodes))
            self.edgelist_to_matrix()
            self.nodedict = self.matrix_to_nodedict()
            self.nodes = range(self.matrix.shape[0])
        if nodedict != {}:
            self.nodedict = nodedict
            self.nodes = range(self.nodedict.keys())
            
    def edgelist_to_matrix(self):
        #Converts the list of tuples representing the edges to a matrix representation of the graph.
        self.matrix = np.zeros((self.nnodes,self.nnodes))
        for edge in self.edges:
            self.matrix[edge[0], edge[1]] = self.edgedict[edge]
            
    def matrix_to_edgelist(self):
        #Converts the matrix representation to a list of tuples representing the edges.
        #unnecessary?
        ind = np.nonzero(self.matrix)
        fnodes, tnodes = ind[0].astype(int), ind[1].astype(int)
        self.edges =  zip(fnodes, tnodes)

    def matrix_to_nodedict(self):
        nodedict = {}
        for i in range(self.nnodes):
            ind = np.nonzero(self.matrix[i,:])[0].tolist()
            nodedict[i] = ind
        return nodedict
    
    def nodedict_to_edgelist(self):
        self.edges = []
        for start in self.nodedict.keys():
            for end in self.nodedict[start]:
                self.edges.append((start, end))
        
    def add_node(self, name, connections):
        #name can be number, connections as list
        self.nodedict[node] = connections
        #update other representations
        
    def add_edge(self, fnode, tnode):
        self.nodedict[fnode].append(tnode)
        #update other representations
        
    def remove_node(self, node):
        assert node in self.nodes, "Node not in graph"
        self.nodedict.pop(node, None)
        #update other representations
        
    def remove_edge(self, node):
        0
        
    def number_of_edges(self):
        return sum(sum(self.matrix))/2
        
        
    def neighbours(self, node):
        return self.nodedict[node]
    
    def number_of_neighbours(self, node):
        return len(self.neighbours(node))
        
    def find_path(self, start, end, path=[]):
        assert start in self.nodes, "Start is not a node in the graph"
        assert end in self.nodes, "End is not a node in the graph"
        path = path + [start]
        if start == end:
            return path
        for node in self.nodedict[start]:
            if node not in path:
                newpath = self.find_path(node, end, path)
                if newpath: 
                    return newpath
        return None
    
    def find_all_paths(self, start, end, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        paths = []
        for node in self.nodedict[start]:
            if node not in path:
                newpaths = self.find_all_paths(node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths 
    
    def find_shortest_path(self, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        shortest = None
        for node in self.nodedict[start]:
            if node not in path:
                newpath = self.find_shortest_path(node, end, path)
                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
        return shortest
        
    def diameter(self):
        """ calculates the diameter of the graph """
        pairs = [ (i,j) for i in range(self.nnodes-1) for j in range(i+1, self.nnodes)]
        shortest_paths = []
        for (start,end) in pairs:
            paths = self.find_all_paths(start,end)
            try:
                shortest = sorted(paths, key=len)[0]
                shortest_paths.append(shortest)
            except:
                return "Infinity"
        diameter = len(shortest_paths[-1])
        return diameter
    
    def local_clustering(self, node):
        c = 0
        for i in self.neighbours(node):
            for j in self.neighbours(node):
                if (i, j) in self.edges:
                    c += 1
        try:
            return c/float(self.number_of_neighbours(node)*(self.number_of_neighbours(node)-1))
        except:
            return 0
    
    def average_clustering(self):
        c=0
        for node in self.nodes:
            c += self.local_clustering(node)
        return c/float(len(self.nodes))
  

#Creating different graphs
def create_erdos(n = 1, p = 1.):
    # n Number of Nodes, p Choice to get edge
    # only upper triangular entries
    matrix = np.random.rand(n,n) < p
                    
    #set diagonal and lower triangular entries to zero
    matrix = np.triu(matrix, 1)
            
    return Graph(matrix = matrix, nnodes = n)

def create_watts(n = 1,  k=1, p = 0.):
    matrix = np.zeros((n,n))
    for i in range(n):
        matrix[i,1+i:(i+k/2+1)] = 1
        matrix[i, (i+n-k/2):n] = 1
        
    for i in range(n):
        for j in range(n):
            if matrix[i,j] == 1 and random.random() < p:
                try:
                    matrix[i,random.choice(i+1+np.nonzero(matrix[i,i+1:] == 0)[0])] = 1
                    matrix[i,j] = 0
                except:
                    0
                    
    #also replace edges from other node?
    
    #set diagonal and lower triangular entries to zero
    matrix = np.triu(matrix, 1)
        
    return Graph(matrix = matrix)
    
    
    
def create_barabasi(n, n_0):
    assert n_0 + 1 < n, "%n should be smaller than n_0+2"
    matrix = np.zeros((n,n))
    #connect first node to 2,3...,n_0+1
    matrix[0,1:n_0+1] = 1
    #connect (n_0+2)th to 2,3...,n_0+1  (still not important which)
    matrix[1:n_0+1, n_0+1] = 1
    for new in range(n_0+2, n):
        nodes = range(n)
        nodes.remove(new)
        probs = list(matrix.sum(axis=1)+matrix.sum(axis=0))
        probs.pop(new)
        probs = probs/sum(probs)
        ind = np.random.choice(nodes, size=n_0, replace=False, p=probs)
        matrix[ind, new] = 1
    return Graph(matrix = matrix)
