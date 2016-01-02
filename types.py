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
    assert n_0 + 1 < n, "%n should be greater than n_0+1"
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
        ind = numpy.random.choice(nodes, size=n_0, replace=False, p=probs)
        matrix[ind, new] = 1
    return Graph(matrix = matrix)
