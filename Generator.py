import random
import networkx as nx

density = lambda G : nx.density(G)

class EdgeMarkovianGenerator:
    def __init__(self, p, q, n, T, d0 = -1) -> None:
        self.p = p
        self.q = q
        self.n = n
        self.t, self.T = 0, T
        self.graphs = []
        if p == 0 and q == 0:
             estimated_limit = 0
        elif d0 >= 0:
            print(f"Using d0 = {d0}")
            estimated_limit = d0
        else:estimated_limit = (1-q) / (2 - p - q) if p + q != 2 else random.random()
        self.graphs.append(nx.random_graphs.binomial_graph(n, estimated_limit))

    def getLast(self) -> nx.Graph:
        return self.graphs[-1]
    
    def getDensityMap(self):
        return map(density, self.graphs)
    
    def step(self):
        G = self.graphs[-1]; G : nx.Graph
        nextG = nx.empty_graph(self.n); nextG : nx.Graph
        for i in range(self.n):
            for j in range(i):
                if (G.has_edge(i, j) or G.has_edge(j, i)) and random.random() < self.p:
                        nextG.add_edge(i, j)
                elif (not G.has_edge(i, j) and not G.has_edge(j, i)) and random.random() > self.q:
                        nextG.add_edge(i, j)
        self.graphs.append(nextG)

    def simulate(self):
         for t in range(self.T):
              self.step()