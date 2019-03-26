"""
Created on Tue Oct 23 14:54:17 2018
Load data
@author: vayer
"""
from graph import Graph
import networkx as nx
from utils import read_files,per_section,indices_to_one_hot
from collections import defaultdict
import numpy as np

class NotImplementedError(Exception):
    pass

def load_local_data(data_path,name,one_hot=False,attributes=True,use_node_deg=False):
    if name=='mutag':
        path=data_path+'/MUTAG_2/'
        dataset=build_MUTAG_dataset(path,one_hot=one_hot)
    if name=='ptc':
        path=data_path+'/PTC_MR/'
        dataset=build_PTC_dataset(path,one_hot=one_hot)
    if name=='nci1':
        path=data_path+'/NCI1/'
        if one_hot==True:
            raise NotImplementedError
        dataset=build_NCI1_dataset(path)
    if name=='imdb-b':
        path=data_path+'/IMDB-BINARY/'
        dataset=build_IMDB_dataset(path,s='BINARY',use_node_deg=use_node_deg)
    if name=='imdb-m':
        path=data_path+'/IMDB-MULTI/'
        dataset=build_IMDB_dataset(path,s='MULTI',use_node_deg=use_node_deg)
    if name=='enzymes':
        path=data_path+'/ENZYMES_2/'
        if attributes:
            dataset=build_ENZYMES_dataset(path,type_attr='real')
        else:
            dataset=build_ENZYMES_dataset(path)
    if name=='protein':
        path=data_path+'/PROTEINS_full/'
        if attributes:
            dataset=build_PROTEIN_dataset(path,type_attr='real',use_node_deg=use_node_deg)
        else:
            dataset=build_PROTEIN_dataset(path)
    if name=='protein_notfull':
        path=data_path+'/PROTEINS/'
        if attributes:
            dataset=build_PROTEIN2_dataset(path,type_attr='real',use_node_deg=use_node_deg)
        else:
            dataset=build_PROTEIN2_dataset(path)
    if name=='bzr':
        path=data_path+'/BZR/'
        if attributes:
            dataset=build_BZR_dataset(path,type_attr='real',use_node_deg=use_node_deg)
        else:
            dataset=build_BZR_dataset(path)
    if name=='cox2':
        path=data_path+'/COX2/'
        if attributes:
            dataset=build_COX2_dataset(path,type_attr='real',use_node_deg=use_node_deg)
        else:
            dataset=build_COX2_dataset(path)
    if name=='synthetic':
        path=data_path+'/SYNTHETIC/'
        if attributes:
            dataset=build_SYNTHETIC_dataset(path,type_attr='real')
        else:
            dataset=build_SYNTHETIC_dataset(path)
    if name=='aids':
        path=data_path+'/AIDS/'
        if attributes:
            dataset=build_AIDS_dataset(path,type_attr='real')
        else:
            dataset=build_AIDS_dataset(path)
    if name=='cuneiform':
        path=data_path+'/Cuneiform/'
        if attributes:
            dataset=build_Cuneiform_dataset(path,type_attr='real')
        else:
            dataset=build_Cuneiform_dataset(path)
    if name=='letter_high':
        path=data_path+'/Letter-high/'
        if attributes:
            dataset=build_LETTER_dataset(path,type_attr='real',name='high')
        else:
            dataset=build_LETTER_dataset(path,name='med')
    if name=='letter_med':
        path=data_path+'/Letter-med/'
        if attributes:
            dataset=build_LETTER_dataset(path,type_attr='real',name='med')
        else:
            dataset=build_LETTER_dataset(path,name='med') 
    if name=='fingerprint':
        path=data_path+'/Fingerprint/'
        dataset=build_Fingerprint_dataset(path,type_attr='real')
    return dataset

#%%
def generate_binary_uniform_tree(maxdepth,coupling='cross',a=0,b=5,c=5,d=10):#il faut que nlowLeaves soit une puissance de 2
    graph=Graph()
    #randint=np.random.randint(2,high=maxdepth)
    randint=maxdepth
    nlowLeaves=2**randint
    groupe=('A',a,b)
    graph.create_classes_uniform_leaves(nlowLeaves,groupe)
    groupe=('B',c,d)
    graph.create_classes_uniform_leaves(nlowLeaves,groupe)
    noeud_0=graph.find_leaf('A')
    noeud_1=graph.find_leaf('B')
    if coupling=='cross':            
        k=0
        for noeud in noeud_0:
            graph.binary_link(noeud,noeud_1[k])
            k=k+1
    else :
        graph.iterative_binary_link(noeud_0,maxIter=1)
        graph.iterative_binary_link(noeud_1,maxIter=1)
    otherNode=list(set(graph.nodes()).difference(set(noeud_1).union(set(noeud_0))))

    graph.iterative_binary_link(otherNode)
    graph.nx_graph=nx.relabel_nodes(graph.nx_graph,{max(graph.nodes(), key=len):1}) #renomer la racine
    graph.construct_tree()

    return graph

def build_binary_uniform_dataset(nTree1,nTree2,maxdepth,a=0,b=5,c=5,d=10):
    data=[]
    for i in range(nTree1):
        data.append((generate_binary_uniform_tree(maxdepth,coupling='cross',a=a,b=b,c=c,d=d),0))
    for i in range(nTree2):
        data.append((generate_binary_uniform_tree(maxdepth,coupling='nocross',a=a,b=b,c=c,d=d),1))

    return data

def build_one_tree_dataset_from_xml(path,classe,max_depth):
    onlyfiles = read_files(path)
    data=[]
    for f in onlyfiles :
        G=Graph()
        G.build_Xml_tree(path+'/'+f,max_depth)
        data.append((G,classe)) 

    return data

def node_labels_dic(path,name):
    node_dic=dict()
    with open(path+name) as f:
        sections = list(per_section(f))
        k=1
        for elt in sections[0]:
            node_dic[k]=int(elt)
            k=k+1
    return node_dic

def node_attr_dic(path,name):
    node_dic=dict()
    with open(path+name) as f:
        sections = list(per_section(f))
        k=1
        for elt in sections[0]:
            node_dic[k]=[float(x) for x in elt.split(',')]
            k=k+1
    return node_dic

def graph_label_list(path,name):
    graphs=[]
    with open(path+name) as f:
        sections = list(per_section(f))
        k=1
        for elt in sections[0]:
            graphs.append((k,int(elt)))
            k=k+1
    return graphs
    
def graph_indicator(path,name):
    data_dict = defaultdict(list)
    with open(path+name) as f:
        sections = list(per_section(f))
        k=1
        for elt in sections[0]:
            data_dict[int(elt)].append(k)
            k=k+1
    return data_dict

def compute_adjency(path,name):
    adjency= defaultdict(list)
    with open(path+name) as f:
        sections = list(per_section(f))
        for elt in sections[0]:
            adjency[int(elt.split(',')[0])].append(int(elt.split(',')[1]))
    return adjency


def all_connected(X):
    a=[]
    for graph in X:
        a.append(nx.is_connected(graph.nx_graph))
    return np.all(a)


def build_NCI1_dataset(path):
    node_dic=node_labels_dic(path,'NCI1_node_labels.txt')
    node_dic2={}
    for k,v in node_dic.items():
        node_dic2[k]=v-1
    node_dic=node_dic2
    graphs=graph_label_list(path,'NCI1_graph_labels.txt')
    adjency=compute_adjency(path,'NCI1_A.txt')
    data_dict=graph_indicator(path,'NCI1_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data

def build_PROTEIN_dataset(path,type_attr='label',use_node_deg=False):
    if type_attr=='label':
        node_dic=node_labels_dic(path,'PROTEINS_full_node_labels.txt') # A voir pour les attributes
    if type_attr=='real':
        node_dic=node_attr_dic(path,'PROTEINS_full_node_attributes.txt')
    graphs=graph_label_list(path,'PROTEINS_full_graph_labels.txt')
    adjency=compute_adjency(path,'PROTEINS_full_A.txt')
    data_dict=graph_indicator(path,'PROTEINS_full_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if not use_node_deg:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        data.append((g,i[1]))

    return data
    
def build_PROTEIN2_dataset(path,type_attr='label',use_node_deg=False):
    if type_attr=='label':
        node_dic=node_labels_dic(path,'PROTEINS_node_labels.txt') # A voir pour les attributes
    if type_attr=='real':
        node_dic=node_attr_dic(path,'PROTEINS_node_attributes.txt')
    graphs=graph_label_list(path,'PROTEINS_graph_labels.txt')
    adjency=compute_adjency(path,'PROTEINS_A.txt')
    data_dict=graph_indicator(path,'PROTEINS_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if not use_node_deg:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        data.append((g,i[1]))

    return data

def build_MUTAG_dataset(path,one_hot=False):
    graphs=graph_label_list(path,'MUTAG_graph_labels.txt')
    adjency=compute_adjency(path,'MUTAG_A.txt')
    data_dict=graph_indicator(path,'MUTAG_graph_indicator.txt')
    node_dic=node_labels_dic(path,'MUTAG_node_labels.txt') # ya aussi des nodes attributes ! The fuck ?
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if one_hot:
                attr=indices_to_one_hot(node_dic[node],7)
                g.add_one_attribute(node,attr)
            else:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data


def build_IMDB_dataset(path,s='MULTI',use_node_deg=False):
    graphs=graph_label_list(path,'IMDB-'+s+'_graph_labels.txt')
    adjency=compute_adjency(path,'IMDB-'+s+'_A.txt')
    data_dict=graph_indicator(path,'IMDB-'+s+'_graph_indicator.txt')
    #node_dic=node_labels_dic(path,'MUTAG_node_labels.txt') # ya aussi des nodes attributes ! The fuck ?
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            #g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        data.append((g,i[1]))
        
    

    return data

def build_PTC_dataset(path,one_hot=False):
    graphs=graph_label_list(path,'PTC_MR_graph_labels.txt')
    adjency=compute_adjency(path,'PTC_MR_A.txt')
    data_dict=graph_indicator(path,'PTC_MR_graph_indicator.txt')
    node_dic=node_labels_dic(path,'PTC_MR_node_labels.txt') # ya aussi des nodes attributes ! The fuck ?
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if one_hot:
                attr=indices_to_one_hot(node_dic[node],18)
                g.add_one_attribute(node,attr)
            else:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data


def build_ENZYMES_dataset(path,type_attr='label',use_node_deg=False):
    graphs=graph_label_list(path,'ENZYMES_graph_labels.txt')
    if type_attr=='label':
        node_dic=node_labels_dic(path,'ENZYMES_node_labels.txt') # A voir pour les attributes
    if type_attr=='real':
        node_dic=node_attr_dic(path,'ENZYMES_node_attributes.txt')
    adjency=compute_adjency(path,'ENZYMES_A.txt')
    data_dict=graph_indicator(path,'ENZYMES_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if not use_node_deg:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        data.append((g,i[1]))

    return data

def build_BZR_dataset(path,type_attr='label',use_node_deg=False):
    graphs=graph_label_list(path,'BZR_graph_labels.txt')
    if type_attr=='label':
        node_dic=node_labels_dic(path,'BZR_node_labels.txt') # A voir pour les attributes
    if type_attr=='real':
        node_dic=node_attr_dic(path,'BZR_node_attributes.txt')
    adjency=compute_adjency(path,'BZR_A.txt')
    data_dict=graph_indicator(path,'BZR_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if not use_node_deg:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        data.append((g,i[1]))

    return data

def build_COX2_dataset(path,type_attr='label',use_node_deg=False):
    graphs=graph_label_list(path,'COX2_graph_labels.txt')
    if type_attr=='label':
        node_dic=node_labels_dic(path,'COX2_node_labels.txt') # A voir pour les attributes
    if type_attr=='real':
        node_dic=node_attr_dic(path,'COX2_node_attributes.txt')
    adjency=compute_adjency(path,'COX2_A.txt')
    data_dict=graph_indicator(path,'COX2_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if not use_node_deg:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        data.append((g,i[1]))

    return data
    
def build_SYNTHETIC_dataset(path,type_attr='label'):
    graphs=graph_label_list(path,'SYNTHETIC_graph_labels.txt')
    if type_attr=='label':
        node_dic=node_labels_dic(path,'SYNTHETIC_node_labels.txt') # A voir pour les attributes
    if type_attr=='real':
        node_dic=node_attr_dic(path,'SYNTHETIC_node_attributes.txt')
    adjency=compute_adjency(path,'SYNTHETIC_A.txt')
    data_dict=graph_indicator(path,'SYNTHETIC_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data
    
def build_AIDS_dataset(path,type_attr='label'):
    graphs=graph_label_list(path,'AIDS_graph_labels.txt')
    if type_attr=='label':
        node_dic=node_labels_dic(path,'AIDS_node_labels.txt') # A voir pour les attributes
    if type_attr=='real':
        node_dic=node_attr_dic(path,'AIDS_node_attributes.txt')
    adjency=compute_adjency(path,'AIDS_A.txt')
    data_dict=graph_indicator(path,'AIDS_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data
    
def build_Cuneiform_dataset(path,type_attr='label'):
    graphs=graph_label_list(path,'Cuneiform_graph_labels.txt')
    if type_attr=='label':
        node_dic=node_labels_dic(path,'Cuneiform_node_labels.txt') # A voir pour les attributes
    if type_attr=='real':
        node_dic=node_attr_dic(path,'Cuneiform_node_attributes.txt')
    adjency=compute_adjency(path,'Cuneiform_A.txt')
    data_dict=graph_indicator(path,'Cuneiform_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data
    
def build_LETTER_dataset(path,type_attr='label',name='med'):
    graphs=graph_label_list(path,'Letter-'+name+'_graph_labels.txt')
    if type_attr=='label':
        node_dic=node_labels_dic(path,'Letter-'+name+'_node_labels.txt') # A voir pour les attributes
    if type_attr=='real':
        node_dic=node_attr_dic(path,'Letter-'+name+'_node_attributes.txt')
    adjency=compute_adjency(path,'Letter-'+name+'_A.txt')
    data_dict=graph_indicator(path,'Letter-'+name+'_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data

def build_Fingerprint_dataset(path,type_attr='real'):
    graphs=graph_label_list(path,'Fingerprint_graph_labels.txt')
    node_dic=node_attr_dic(path,'Fingerprint_node_attributes.txt')
    adjency=compute_adjency(path,'Fingerprint_A.txt')
    data_dict=graph_indicator(path,'Fingerprint_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data
