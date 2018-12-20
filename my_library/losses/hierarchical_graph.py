from collections import defaultdict
from tqdm import tqdm

import igraph

class HierarchicalGraph():
    """ Graph data structure, undirected by default. """

    def __init__(self, connections=None, directed=False):
        self._graph = defaultdict(set)
        self._nodes = defaultdict(set)
        self._directed = directed
        if connections:
            self.add_connections(connections)

    def add_connections(self, connections):
        """ Add connections (list of tuple pairs) to graph """

        for node1, node2 in connections:
            self.add(node1, node2)

    def add(self, node1, node2):
        """ Add connection between node1 and node2 """

        self._graph[node1].add(node2)
        if not self._directed:
            self._graph[node2].add(node1)
            
        self._nodes[node2].add(node1)

    def remove(self, node):
        """ Remove all references to node """

        for n, cxns in self._graph.iteritems():
            try:
                cxns.remove(node)
            except KeyError:
                pass
        try:
            del self._graph[node]
        except KeyError:
            pass

    def is_connected(self, node1, node2):
        """ Is node1 directly connected to node2 """

        return node1 in self._graph and node2 in self._graph[node1]

    def get_parents(self, node):
        """ Find all the parents associated with node """
        
        return [n for n in self._graph if node in self._graph[n]]
    
    def get_childs(self, node):
        """ Find all the childs associated with node """
        if node not in self._graph:
            return None
        
        return list(self._graph[node])
    
    def find_path(self, node1, node2, path=[]):
        """ Find any path between node1 and node2 (may not be shortest) """

        path = path + [node1]
        if node1 == node2:
            return path
        if node1 not in self._graph:
            return None
        for node in self._graph[node1]:
            if node not in path:
                new_path = self.find_path(node, node2, path)
                if new_path:
                    return new_path
        return None
    
    def find_shortest_path(self, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        if start not in self._graph:
            return None
        shortest = None
        for node in self._graph[start]:
            if node not in path:
                new_path = self.find_shortest_path(node, end, path)
                if new_path:
                    if not shortest or len(new_path) < len(shortest):
                        shortest = new_path
        return shortest
    
    def get_shortest_possible_path(self, node):
        """ Find shortest possible path of node """
        
        if node in self._roots:
            return [node]
        
        if node not in self._nodes:
            return [node]
        
        candidate_paths = [self.find_shortest_path(n, node) for n in self._roots]
        candidate_paths = [p for p in candidate_paths if p]
        
        if candidate_paths:
            return min(candidate_paths, key=len)
        
        return None
    
    def get_degree(self, node1, node2):
        """ Find the degree between node1 and node2 (i.e. length of shortest path)"""        
        shortest_path_f = self.find_shortest_path(node1, node2)
        shortest_path_b = self.find_shortest_path(node2, node1)
        
        return len(shortest_path_f or shortest_path_b or [])
    
    def load(self, connections):
        self.add_connections(connections['link'])
        self._root_n_leaf = connections['root_n_leaf']
        self._roots = [n for n in tqdm(self._graph) if len(self._nodes[n]) == 0]
    
    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))

    
class HGraph():
    """ Graph data structure, undirected by default. """

    def __init__(self, connections=None, directed=True):
        self._graph = igraph.Graph(directed=True)
        self._node2idx = {}
        self._idx2node = {}
        self._curr_idx = 0
        if connections:
            self.add_connections(connections)

    def add_connections(self, connections):
        """ Add connections (list of tuple pairs) to graph """
        
        for i,j in connections:
            if i not in self._node2idx:
                self._node2idx[i] = self._curr_idx
                self._idx2node[self._curr_idx] = i
                self._curr_idx += 1
            if j not in self._node2idx:
                self._node2idx[j] = self._curr_idx
                self._idx2node[self._curr_idx] = j
                self._curr_idx += 1
        
        self._graph.add_vertices(len(self._node2idx))
        self._graph.add_edges([(self._node2idx[i],self._node2idx[j]) for i,j in connections])

        self._roots = [n for n in self._node2idx if not self.get_parents(n)]
        
    def is_connected(self, node1, node2):
        """ Is node1 directly connected to node2 """
        	
        return self._graph.are_connected(self._node2idx[node1],self._node2idx[node2])

    def get_parents(self, node):
        """ Find all the parents associated with node """
        
        return [self._idx2node[n] for n in self._graph.predecessors(self._node2idx[node])]
    
    def get_childs(self, node):
        """ Find all the childs associated with node """
        
        return [self._idx2node[n] for n in self._graph.successors(self._node2idx[node])]

    def find_shortest_path(self, node1, node2):
        """ Find shortest path between node1 and node2 """

        path = [self._idx2node[i] for i in self._graph.get_shortest_paths(self._node2idx[node1], self._node2idx[node2])[-1]]
        
        if path:
            return path
        
        return None
    
    def get_shortest_possible_path(self, node):
        """ Find shortest possible path of node """
        
        if node in self._roots:
            return None
        
        candidate_paths = [self.find_shortest_path(n, node) for n in self._roots]
        candidate_paths = [p for p in candidate_paths if p]
        
        if candidate_paths:
            return min(candidate_paths, key=len)
        
        return None
    
    def get_degree(self, node1, node2):
        """ Find the degree between node1 and node2 (i.e. length of shortest path)"""        
        shortest_path_f = self.find_shortest_path(node1, node2)
        shortest_path_b = self.find_shortest_path(node2, node1)
        
        return len(shortest_path_f or shortest_path_b or [])
    
    def load(self, connections):
        self.add_connections(connections)
