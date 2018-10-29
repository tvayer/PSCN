#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

def load_local_data(data_path,name,one_hot=False,attributes=False):
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
        dataset=build_IMDB_dataset(path,s='BINARY')
    if name=='imdb-m':
        path=data_path+'/IMDB-MULTI/'
        dataset=build_IMDB_dataset(path,s='MULTI')
    if name=='enzymes':
        path=data_path+'/ENZYMES_2/'
        if attributes:
            dataset=build_ENZYMES_dataset(path,type_attr='real')
        else:
            dataset=build_ENZYMES_dataset(path)
    if name=='protein':
        path=data_path+'/PROTEINS_full/'
        if attributes:
            dataset=build_PROTEIN_dataset(path,type_attr='real')
        else:
            dataset=build_PROTEIN_dataset(path)
    if name=='bzr':
        path=data_path+'/BZR/'
        if attributes:
            dataset=build_BZR_dataset(path,type_attr='real')
        else:
            dataset=build_BZR_dataset(path)
    if name=='cox2':
        path=data_path+'/COX2/'
        if attributes:
            dataset=build_COX2_dataset(path,type_attr='real')
        else:
            dataset=build_COX2_dataset(path) 
    return dataset

#%%

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

def build_PROTEIN_dataset(path,type_attr='label'):
    if type_attr=='label':
        node_dic=node_labels_dic(path,'PROTEINS_full_node_labels.txt') 
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
            g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data

def build_MUTAG_dataset(path,one_hot=False):
    graphs=graph_label_list(path,'MUTAG_graph_labels.txt')
    adjency=compute_adjency(path,'MUTAG_A.txt')
    data_dict=graph_indicator(path,'MUTAG_graph_indicator.txt')
    node_dic=node_labels_dic(path,'MUTAG_node_labels.txt') 
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


def build_random_MUTAG_dataset(size=10):
    data=[]
    attributes_possibility=[0,1,2,3,4,5,6]
    for i in range(size):
        g=Graph()
        node_number = np.random.randint(3,55)
        G = nx.barabasi_albert_graph(node_number, 2)
        g.nx_graph=G
        for node in g.nodes():
            g.add_one_attribute(node,np.random.choice(attributes_possibility))
        data.append((g,-10))
    return data

def build_IMDB_dataset(path,s='MULTI'):
    graphs=graph_label_list(path,'IMDB-'+s+'_graph_labels.txt')
    adjency=compute_adjency(path,'IMDB-'+s+'_A.txt')
    data_dict=graph_indicator(path,'IMDB-'+s+'_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data

def build_PTC_dataset(path,one_hot=False):
    graphs=graph_label_list(path,'PTC_MR_graph_labels.txt')
    adjency=compute_adjency(path,'PTC_MR_A.txt')
    data_dict=graph_indicator(path,'PTC_MR_graph_indicator.txt')
    node_dic=node_labels_dic(path,'PTC_MR_node_labels.txt') 
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


def build_ENZYMES_dataset(path,type_attr='label'):
    graphs=graph_label_list(path,'ENZYMES_graph_labels.txt')
    if type_attr=='label':
        node_dic=node_labels_dic(path,'ENZYMES_node_labels.txt') 
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
            g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data

def build_BZR_dataset(path,type_attr='label'):
    graphs=graph_label_list(path,'BZR_graph_labels.txt')
    if type_attr=='label':
        node_dic=node_labels_dic(path,'BZR_node_labels.txt') 
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
            g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data

def build_COX2_dataset(path,type_attr='label'):
    graphs=graph_label_list(path,'COX2_graph_labels.txt')
    if type_attr=='label':
        node_dic=node_labels_dic(path,'COX2_node_labels.txt') 
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
            g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data


