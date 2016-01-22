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
        
    #also for edgedict?

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
        
    def remove_fraction_edges(self, fraction):
        matrix = np.triu(self.matrix, 1)
        i,j = np.nonzero(matrix)
        ix = np.random.choice(len(i), np.floor(fraction * len(i)), replace=False)
        matrix[i[ix], j[ix]] = 0
        self.matrix = matrix + matrix.T - np.diag(matrix.diagonal())
        self.update_graph(from_rep = "matrix")
        
    def remove_node(self, node):
        assert node in self.nodes, "Node not in graph"
        self.matrix[node,:] = 0
        self.matrix[:, node] = 0
        self.nodes.remove(node)
        #update other representations, if only working with nodedict/paths not necessary
        self.update_graph(from_rep = "matrix")
        
    
    def remove_edge(self, node_i, node_j):
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
    
    def list_degrees(self):
        lsd = []
        for i in self.nodes:
            lsd.append(self.degree(i))
        return lsd

    def degree_matrix(self):
        dmatrix = np.zeros((self.nnodes, self.nnodes))
        for i in self.node:
            dmatrix[i,i] = self.degree(i)
        return dmatrix
    
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
    
    def dijkstra_node(self, node):
        #calculates all distances from node with Dijkstra
        #what if graph is not connected?
        dist_dict = dict(zip(self.nodes, [float("infinity")]*self.nnodes))
        dist_dict[node] = 0
        current = node
        unvisited = copy.deepcopy(self.nodes)
        while unvisited != []:
            for neighbour in self.neighbours(current): #exclude visited nodes
                distance = self.edgedict[(current, neighbour)] + dist_dict[current]
                if distance < dist_dict[neighbour]:
                    dist_dict[neighbour] = distance
            poss = [pn for pn, value in dist_dict.items() if value != float("infinity")]
            if True:
                #the next node has to have dist_dict value != infinity
                #and must be in unvisited
                unvisited.remove(current)
                poss = [pn for pn, value in dist_dict.items() if value != float("infinity")]
                next_node = random.choice(list(set(poss) & set(unvisited)))
                current = next_node
                if len(unvisited) == 1:
                    next_node = unvisited[0]
                    unvisited = []
                    current = next_node
            #if nodes that have dist_dict value == infinity
            #equals unvisited then there are nodes that are not connected
            #raise UnconnectednessError
        print dist_dict
        return dist_dict[max(dist_dict.values())], max(dist_dict.values())
    
    def distance(self, node_i, node_j):
        return len(self.find_shortest_path(node_i, node_j))
        
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
    
    def average_gedesic_distance(self):
        dist_sum = 0
        for i in self.nodes:
            #sufficient to count from node to avoid counting double
            for j in self.nodes[i+1:]:
                if i != j:   
                    # distance(i,j) = distance(j,i) in undirected network
                    dist_sum += 2*self.distance(i, j)
                    
        return dist_sum/(self.nnodes*(self.nnodes-1))
    
    def max_dist_node(self, node):
        #returns the node farthest wawy from node with the distance
        
        return 0
    
    def fast_diameter(self, max_count = 5):
        """
            We are going to try to find the diameter of the connected part of the network.
           We start with an arbitrary node and find the node that lies furthest away
           from it. Now we find the node for which the sum of the distances to
           both nodes is maximal - and, of course, the maximum distance from the
           second node. We repeat the procedure a few times until the no larger
           diameter has been found a few times.
           This yields the diameter of the network with a very high likelihood and
           saves a lot of work compared to finding all distances between all nodes.
        """
        counter = 0
        maxlen = 0
        #select a node
        n1 = random.choice(self.nodes)
        while True:
            n2, ml2 = self.dijkstra_node(n1)
            print "N2", n2, ml2
            n3, ml3 = self.dijkstra_node(n2)
            print "N3", n3, ml3
            if n3 == n1:
                return ml2
            if ml2 == ml3:
                counter += 1
            if counter == max_count:
                return ml2
            n1 = n3
    
    def local_clustering(self, node):
        #determines how well the neighbors of a node are connected
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
        #Averages local clustering for all nodes
        c=0
        for node in self.nodes:
            c += self.local_clustering(node)
        return c/float(len(self.nodes))
    
    def global_clustering(self):
        
        return 0
    
    
    
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
            colorlist = ['r' if i in np.nonzero(inf_list)[0] else 'b' if
                         i in np.nonzero(recovered_list)[0] else 'y' for i in range(self.nnodes)]
            plt.ion()
            G = nx.Graph(self.matrix)
            pos=nx.spring_layout(G)
            ##         #draw sick nodes red, recovered nodes blue, susceptible nodes yellow 
            nx.draw_networkx_nodes(G,pos,
                           node_color=colorlist,
                           node_size=50, alpha=0.8)
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
                colorlist = ['r' if i in np.nonzero(inf_list)[0] else 'b' 
                             if i in np.nonzero(recovered_list)[0] else 'y' for i in range(self.nnodes)]
                nx.draw_networkx_nodes(G,pos,
                                       node_color=colorlist,
                                       node_size=50, alpha=0.8)
                nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
                plt.axis('off')
                fig_title = "SIR, time=%d" %t
#                 plt.savefig(fig_title)
                plt.show()
                sleep(1.)
                plt.close()
            
            
                    
            #update lists
            num_inf = sum(inf_list)
            num_rec = sum(recovered_list)
            susceptible.append(self.nnodes - num_inf - num_rec)
            infected.append(sum(inf_list))    
            recovered.append(num_rec)
            
            #stop when no infected
            if np.sum(inf_list) == 0:
                return susceptible, infected, recovered
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
        while self.is_connected() and len(self.nodes) != 1:
            self.remove_node(self.find_node_highest_degree())
            num_failure += 1
        return num_failure
    
    def distribute_weights(self, random = True):
        #sets weights of edges
        #TODO: implement weight to give as argument
        0
        
    def set_negative_weights(self, fraction = 0.5):
        i,j = np.nonzero(self.matrix)
        ix = np.random.choice(len(i), np.floor(fraction * len(i)), replace=False)
        self.matrix[i[ix], j[ix]] = -1
        
    def set_negative_nodes(self, fraction = 0.1):
        self.neg_nodes = np.random.choice(self.nodes, np.floor(fraction*self.nnodes))
        for n in self.neg_nodes:
            self.matrix[n,:][np.nonzero(self.matrix[n,:])] = -1
#             self.matrix[:,n][np.nonzero(self.matrix[:,n])] = -1
        
    def neural_failure(self, time = 1000, activated = 1, act_threshold = 3, act_time = 2):
        num_failure = 0
        for i in range(self.nnodes):
            num_failure += 1
            self.remove_node(self.find_node_highest_degree())
            sact, aact = self.neural(time = time, activated = activated, act_threshold = act_threshold, act_time = act_time)
            if len(sact) < time:
                return num_failure
        return "Did not fail"
    
    def neural_attack(self, time = 1000, activated = 1, act_threshold = 3, act_time = 2):
        num_failure = 0
        for i in range(self.nnodes):
            num_failure += 1
            self.remove_node(random.choice(self.nodes))
            sact, aact = self.neural(time = time, activated = activated, act_threshold = act_threshold, act_time = act_time)
            if len(sact) < time:
                return num_failure
        return "Did not fail"
        
    
    def neural(self, time, activated = 1, act_threshold = 3, act_time = 2, vis = False):
        """
        Simulates neural activity
        Returns the number of neurons activated at one time for each timestep
        Should return activity of all neurons at each time
        """
        #initialize activations (randomly)
        activations = np.zeros((self.nnodes,1))
        activations[:activated] = 1
        activations = np.random.permutation(activations)
#         print activations
        num_act = [np.sum(activations)]
        act_times = np.zeros((self.nnodes,1))
        all_act = np.zeros((self.nnodes, time))
        if vis:
            colorlist = ['r' if i in np.nonzero(activations)[0] else 'b' for i in range(self.nnodes)]
            plt.ion()
            G = nx.Graph(self.matrix)
            pos=nx.spring_layout(G)
            ##         #draw sick nodes red, recovered nodes blue, susceptible nodes yellow 
            nx.draw_networkx_nodes(G,pos,
                           node_color=colorlist,
                           node_size=50, alpha=0.8)
            nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
            plt.axis('off')
            plt.draw()
            sleep(1.)
        for t in range(time):
            act_times += 1
            #Brute Force update
#             for node in self.nodes:
#                 act = 0
#                 for nei in self.neighbours(node):
#                     if activations[nei] == 1:
#                         act += self.edgedict[(node, nei)]
#                 print act, np.exp(-act), random.random()*np.exp(-act_threshold)
#                 if np.exp(-act) < random.random()*act_threshold:
#                     activations[node] = 1
#                     act_times[node] = 0
#                 activations[act_times >= act_time] = 0
            act_val = self.matrix.dot(activations)
            sact = np.sum(activations)
            #inhibit when many neurons are active, 
            #i.e. decrease chance to fire with increasing overall activity
            activations[np.exp(-act_val)/(1+act_val/(1+sact)) < random.random()*np.exp(-act_threshold)] = 1
            act_times[np.exp(-act_val)/(1+act_val/(1+sact)) < random.random()*np.exp(-act_threshold)] = 0            
            activations[act_times >= act_time] = 0   
            num_act.append(sact)
            all_act[:, t] = activations.reshape(self.nnodes)
            
            if vis:
                colorlist = ['r' if i in np.nonzero(activations)[0] else 'b' for i in range(self.nnodes)]
                nx.draw_networkx_nodes(G,pos,
                                       node_color=colorlist,
                                       node_size=50, alpha=0.8)
                nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
                plt.axis('off')
                fig_title = "SIR, time=%d" %t
#                 plt.savefig(fig_title)
                plt.show()
                sleep(1.)
                plt.close()
            
            if sact == 0:
                return num_act, all_act
        return num_act, all_act
    
    def izhikevich(self, time, frac_neg = 0.1, record = 1):
        #set part of neurons to inhibitory
        self.set_negative_nodes(fraction = frac_neg)
        neg_num = np.floor(frac_neg*self.nnodes)
#         print neg_num, len(self.neg_nodes)
        
        #random distributions
        rall = np.random.rand(self.nnodes,1)
        re = np.random.rand(self.nnodes-neg_num,1)
        ri = np.random.rand(neg_num,1)
        
        #set up parameters
        a = 0.02*np.ones((self.nnodes,1))
        a[self.neg_nodes] = 0.02+0.08*ri
        
        b = 0.2*np.ones((self.nnodes,1))
        b[self.neg_nodes] = 0.25-0.05*ri
        
        c =  -65+15*rall**2
        c[self.neg_nodes] = -65*np.ones((neg_num,1))
        
        d = 8-6*rall**2
        d[self.neg_nodes] = 2*np.ones((neg_num,1))
        
        #S=[0.5*rand(Ne+Ni,Ne), -rand(Ne+Ni,Ni)];
        #should multiply wights with random number?
        self.matrix = self.matrix * np.random.rand(self.nnodes, self.nnodes)
        
        v = -65*np.ones((self.nnodes,1)) # Initial values of v
        u = b*v                  
        all_act = np.zeros((self.nnodes, time))                    # spike timings
                        
                        
        recorded_act = np.zeros((record, time))
        
        #simulation
        for t in range(time):
            #random (sensory) input
            I=5*np.random.rand(self.nnodes,1)
            I[self.neg_nodes] = np.random.rand(neg_num,1) # thalamic input
            
            fired = np.where(v>=30)[0]

            all_act[fired,t] = 1
            
            v[fired]=c[fired]
            u[fired]=u[fired]+d[fired]
            
            try:
                I = I + np.sum(self.matrix[:,fired],2)
            except:
                0
            
            v=v+0.5*(0.04*v**2+5*v+140-u+I)  #  step 0.5 ms for numerical stability
            v=v+0.5*(0.04*v**2+5*v+140-u+I)   
            u=u+a*(b*v-u)               
            
            if record > 0:
                recorded_act[:, t] = v[:record]

        return all_act, recorded_act
    
    

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


#other functions
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]

def acf(x, length=5000):
    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1] for i in range(1, length)])

def variance_matrix(binned_data):
    r, c =  binned_data.shape
    varmat = np.zeros((r,r))
    for i in range(r):
        vari = np.var(binned_data[i,:])
        for j in range(r):
            varmat[i,j] = 1/np.sqrt(vari*np.var(binned_data[j,:]))
            
    return varmat

def bin_data(data):
    time_matrix = np.zeros((302,10))
    for i in range(10):
        time_matrix[:,i] = np.sum(data[:,i*500:i*500+500],1)
    return 

def firing_correlation(data):
    bd = bin_data(data)
    vm = variance_matrix(bd)
    cov = np.cov(bd)
    return cov*vm
