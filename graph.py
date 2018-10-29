""" A Python Class

Compatible networkx VERSION 2
"""
import networkx as nx
import numpy as np
import time


class NoAttrMatrix(Exception):
    pass

class NoPathException(Exception):
    pass


class Graph():

    def __init__(self):
        self.nx_graph=nx.Graph()
        self.name='A graph as no name'

    def __eq__(self, other) : 
        #print('yo method')
        return self.nx_graph == other.nx_graph

    def __hash__(self):
        return hash(str(self))

    def nodes(self):
        return dict(self.nx_graph.nodes())

    def edges(self):
        return self.nx_graph.edges()

    def add_vertex(self, vertex):
        if vertex not in self.nodes():
            self.nx_graph.add_node(vertex)

    def values(self):
        return [v for (k,v) in nx.get_node_attributes(self.nx_graph,'attr_name').items()]

    def add_nodes(self, nodes):
        self.nx_graph.add_nodes_from(nodes)

    def add_edge(self, edge):
        (vertex1, vertex2) = tuple(edge)
        self.nx_graph.add_edge(vertex1,vertex2)

    def add_one_attribute(self,node,attr,attr_name='attr_name'):
        self.nx_graph.add_node(node,attr_name=attr)

    def add_attibutes(self,attributes):
        attributes=dict(attributes)
        for node,attr in attributes.items():
            self.add_one_attribute(node,attr)

    def get_attr(self,vertex):
        return self.nx_graph.node[vertex]


    def find_leaf(self,beginwith): #assez nulle comme recherche
        nodes=self.nodes()
        returnlist=list()
        for nodename in nodes :
            if str(nodename).startswith(beginwith):
                returnlist.append(nodename)
        return returnlist
    
    def smallest_path(self,start_vertex, end_vertex):
        try:
            pathtime=time.time()
            shtpath=nx.shortest_path(self.nx_graph,start_vertex,end_vertex)
            endpathtime=time.time()
            self.log['pathtime'].append(endpathtime-pathtime)
            return shtpath
        except nx.exception.NetworkXNoPath:
            raise NoPathException('No path between two nodes, graph name : ',self.name)

    def reshaper(self,x):
        try:
            a=x.shape[1]
            return x
        except IndexError:
            return x.reshape(-1,1)

    def all_matrix_attr(self,return_invd=False):
        d=dict((k, v) for k, v in self.nx_graph.node.items())
        x=[]
        invd={}
        try :
            j=0
            for k,v in d.items():
                x.append(v['attr_name'])
                invd[k]=j
                j=j+1
            if return_invd:
                return np.array(x),invd
            else:
                return np.array(x)
        except KeyError:
            raise NoAttrMatrix
            

    






