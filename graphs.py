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
            self.matrix_to_nodedict()
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
            self.matrix_to_nodedict()
            self.nodes = range(self.matrix.shape[0])
        if nodedict != {}:
            self.nodedict = nodedict
            self.nodes = range(self.nodedict.keys())
            
    def edgelist_to_matrix(self):
        #Converts the list of tuples representing the edges to a matrix representation of the graph.
        self.matrix = np.zeros((self.nnodes,self.nnodes))
        for edge in self.edges:
            self.matrix[edge[0], edge[1]] = self.edgedict[edge]
            
    def matrix_to_edgelist(self, edgevalues = []):
        #Converts the matrix representation to a list of tuples representing the edges.
        #unnecessary?
        ind = np.nonzero(self.matrix)
        fnodes, tnodes = ind[0].astype(long), ind[1].astype(long)
#         fnodes, tnodes = ind[0], ind[1]
        self.edges =  zip(fnodes, tnodes)
        if edgevalues == []:
            edgevalues = [1] * len(self.edges)
        self.edgedict = dict(zip(self.edges, edgevalues))

    def matrix_to_nodedict(self):
        self.nodedict = {}
        for i in range(self.nnodes):
            ind = np.nonzero(self.matrix[i,:])[0].tolist()
            self.nodedict[i] = ind
    
    def nodedict_to_edgelist(self):
        self.edges = []
        for start in self.nodedict.keys():
            for end in self.nodedict[start]:
                self.edges.append((start, end))
                
    def update_graph(self, from_rep):
        #Update other representations from from_rep = "matrix", "edgelist" or "nodedict"
        if from_rep == "matrix":
            self.matrix_to_edgelist()
            self.matrix_to_nodedict()
        elif from_rep == "edgelist":
            self.edgelist_to_matrix()
            self.matrix_to_nodedict()
        elif from_rep == "nodedict":
            self.nodedict_to_edgelist()
            self.edgelist_to_matrix()
        
    def add_node(self, name, connections):
        #name can be number, connections as list
        self.nodedict[node] = connections
        #update other representations
        
    def add_edge(self, fnode, tnode):
        self.nodedict[fnode].append(tnode)
        #update other representations
        
    def remove_node(self, node):
        assert node in self.nodes, "Node not in graph"
        self.matrix[node,:] = 0
        self.matrix[:, node] = 0
        self.nodes.remove(node)
        #update other representations, if only working with nodedict/paths not necessary
        self.update_graph(from_rep = "matrix")
    
    def remove_edge(self, node):
        0
        
    def number_of_edges(self):
        return sum(sum(self.matrix))/2
        
        
    def neighbours(self, node):
        return self.nodedict[node]
    
    def degree(self, node):
        return len(self.neighbours(node))
    
    def average_degree(self):
        ad = 0
        for i in self.nodes:
            ad += self.degree(i)
        return ad/float(len(self.nodes))
    
    def is_connected(self):
        #checks if there is a connection between all pairs
        #brute force, should be possible to make it more efficient
        pairs = [ (i,j) for i in self.nodes for j in self.nodes]
        for (start,end) in pairs:
            paths = self.find_all_paths(start,end)
            if paths == []:
                return False
        return True
    
    def find_node_highest_degree(self):
        return np.argmax(self.all_degrees())
        
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
                #Raise exception?
                return float("inf")
        diameter = len(sorted(shortest_paths, key=len)[-1]) - 1
        return diameter
    
    def local_clustering(self, node):
        c = 0
        for i in self.neighbours(node):
            for j in self.neighbours(node):
                if (i, j) in self.edges:
                    c += 1
        try:
            return c/float(self.degree(node)*(self.degree(node)-1))
        except:
            return 0
    
    def average_clustering(self):
        c=0
        for node in self.nodes:
            c += self.local_clustering(node)
        return c/float(len(self.nodes))
    
    def degree_dict(self):
        #returns numer of nodes which have a certain degree for each possible degree as a dict
        dd = {}
        for node in self.nodes:
            degree = self.degree(node)
            if degree in dd.keys():
                dd[degree] += 1
            else:
                dd[degree] = 1
        return dd
    
    def degree_list(self):
        #returns numer of nodes which have a certain degree for each possible degree as a list
        dl = [0]*self.nnodes
        for node in self.nodes:
            dl[self.degree(node)] +=1
        return dl
    
    def all_degrees(self):
        dl = [0]*self.nnodes
        #returns the degree of each node as a list in the order as they are in self.nodes
        for node in self.nodes:
            dl[node] = self.degree(node)
        return dl
    
    def SIR(self, time = 1, n_inf = 1, p = 1., sick_time = 5, vis = False):
        inf_list = np.zeros((self.nnodes,1))
        recovered_list = np.zeros((self.nnodes,1)) #these nodes are immune
        infected_nodes = np.random.choice(range(self.nnodes), size=n_inf, replace=False)
        inf_list[infected_nodes] = 1
        inf_time = np.zeros((self.nnodes,1)) #or list?
        susceptible = [self.nnodes - n_inf]
        infected = [n_inf]
        recovered = [0]
        if vis:
            colorlist = ['r' if i in np.nonzero(inf_list)[0] else 'b' if i in np.nonzero(recovered_list)[0] else 'y' for i in range(self.nnodes)]
            plt.ion()
            G = nx.Graph(self.matrix)
            pos=nx.spring_layout(G)
            ##         #draw sick nodes red, recovered nodes blue, susceptible nodes yellow 
            nx.draw_networkx_nodes(G,pos,
                           node_color=colorlist,
                           node_size=500, alpha=0.8)
            nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
            plt.axis('off')
            plt.draw()
            sleep(1.)


        for t in range(time):
            #update sickness times
            inf_time[np.nonzero(inf_list)] += 1
            #recovery chance
            recovered_list[np.nonzero(inf_time == sick_time)] = 1
            inf_list[np.nonzero(inf_time == sick_time)] = 0
            #update infection vector
            infvec = self.matrix.dot(inf_list)
            #print infvec
            for i,inf_people in enumerate(infvec):
                sick = False
                for j in range(inf_people):
                    if random.random() < p:
                        sick = True
                if sick and recovered_list[i] == 0:
                    inf_list[i] = 1
            if vis:
                colorlist = ['r' if i in np.nonzero(inf_list)[0] else 'b' if i in np.nonzero(recovered_list)[0] else 'y' for i in range(self.nnodes)]
                nx.draw_networkx_nodes(G,pos,
                                       node_color=colorlist,
                                       node_size=500, alpha=0.8)
                nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
                plt.axis('off')
                fig_title = "SIR, time=%d" %t
#                 plt.savefig(fig_title)
                plt.show()
                sleep(1.)
                    
            #update lists
            num_inf = sum(inf_list)
            num_rec = sum(recovered_list)
            susceptible.append(self.nnodes - num_inf - num_rec)
            infected.append(sum(inf_list))    
            recovered.append(num_rec)
        plt.show()
        return susceptible, infected, recovered
    
    def failures(self):
        num_failure = 0
        while self.is_connected() and len(self.nodes) != 1:
           
            self.remove_node(random.choice(self.nodes))
            num_failure += 1
            print num_failure
        return num_failure
        
    
    def attacks(self):
        num_failure = 0
        while self.is_connected() or len(self.nodes) == 1:
           
            self.remove_node(random.choice(self.nodes))
            num_failure += 1
        return num_failure
    
    

#Creating different graphs
def create_erdos(n = 1, p = 1.):
    # n Number of Nodes, p Choice to get edge
    # only upper triangular entries
    matrix = np.random.rand(n,n) < p
                    
    #set diagonal and lower triangular entries to zero
    matrix = np.triu(matrix, 1).astype(int)
            
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
#     matrix[1:n_0+1, n_0+1] = 1
    for new in range(n_0+1, n):
        nodes = range(new)
        #nodes.remove(new)
        probs = list(matrix.sum(axis=1)+matrix.sum(axis=0))[:new]
        #probs.pop(new)
        probs = probs/sum(probs)
        ind = np.random.choice(nodes, size=n_0, replace=False, p=probs)
        matrix[ind, new] = 1
    return Graph(matrix = matrix)
