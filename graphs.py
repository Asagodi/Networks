import numpy as np
import math
import random
import copy

#Create three possibilities to represent graph?
#TODO: create possibility to give nodes names? or just number-name dict?
class Graph(object):
    """
    Input can be as a matrix, list of edges as tuples, dictionary of connections between nodes
    """
    def __init__(self, matrix = [], nnodes = 1, nodes = [], \
                 edges = [], nodevalues = [], edgevalues = [], nodedict = {}):
        if matrix != []:
            self.matrix = matrix
            self.nnodes = matrix.shape[0]
            self.matrix_to_edgelist()
            self.nodedict = self.matrix_to_nodedict()
        elif nodes == []:
            self.edges = edges
            self.nnodes = nnodes
            if edgevalues == []:
                edgevalues = [1] * len(self.edges)
            self.edgedict = dict(zip(self.edges, edgevalues))
            self.matrix = np.zeros((nnodes,nnodes))
            self.edgelist_to_matrix()
            self.nodedict = self.matrix_to_nodedict()
        if nodedict != {}:
            0
            
    def edgelist_to_matrix(self):
        #Convert the list of tuples representing edges to a matrix representation of the graph.
        self.matrix = np.zeros((self.nnodes,self.nnodes))
        for edge in self.edges:
            self.matrix[edge[0], edge[1]] = self.edgedict[edge]
            
    def matrix_to_edgelist(self):
        ind = np.nonzero(self.matrix)
        self.edges =  zip(ind[0], ind[1])
        
    def matrix_to_nodedict(self):
        nodedict = {}
        for i in range(self.nnodes):
            ind = np.nonzero(self.matrix[i,:])[0].tolist()
            ind.extend(np.nonzero(self.matrix[:,i])[0].tolist())
            nodedict[i] = ind
        return nodedict
        
        
    def add_edge(self):
        0
        
    def neighbours(self, node):
        0
        
    def find_path(self, start, end, path=[]):
        #assert
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
            shortest = sorted(paths, key=len)[0]
            shortest_paths.append(shortest)
        diameter = len(shortest_paths[-1])
        return diameter
