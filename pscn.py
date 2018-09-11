"Patchy San"

import networkx as nx
from networkx import convert_node_labels_to_integers
from pynauty.graph import canonical_labeling,NautyGraph
import copy
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Model,Sequential
from keras.layers import Conv1D,Dense,Dropout,Flatten,Activation,Input
import numpy as np

class PSCN():
    def __init__(self,w,s=1,k=10,labeling_procedure_name='betweeness'
                 ,epochs=150,batch_size=25,verbose=0):
    """
    w : the length of the node sequence 
    k : the size of the receptive field
    labeling_procedure_name : name of the labeling procedure, just implemented betweeness
    epochs : epochs for the CNN model 
    batch_size : the batch size for the CNN model
    """
        self.w=w
        self.s=s
        self.k=k
        self.labeling_procedure_name=labeling_procedure_name
        self.epochs=epochs
        self.batch_size=batch_size
        self.verbose=verbose
        self.model = KerasClassifier(build_fn=self.create_model
                                     ,epochs=self.epochs, 
                                     batch_size=self.batch_size, verbose=self.verbose)

        
    def create_model(self):
        # just binary for now
       model=Sequential()
       model.add(Conv1D(filters=16,kernel_size=self.k,strides=self.k,input_shape=(self.w*self.k,1)))
       model.add(Conv1D(filters=8,kernel_size=10,strides=1))
       model.add(Flatten())
       model.add(Dense(128,activation="relu"))
       model.add(Dropout(0.5))
       model.add(Dense(1,activation="sigmoid"))
       model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=["accuracy"])    
       return model
   
    def process_data(self,X,y=None):
        """
        X is a list of networkx graphs
        y is a list of class 
        """
        n=len(X)
        train=[]
        for i in range(n):
           rfMaker=ReceptiveFieldMaker(X[i],w=self.w
                                       ,k=self.k,s=self.s
                                       ,labeling_procedure_name=self.labeling_procedure_name)
           forcnn=rfMaker.make_()
           train.append(np.array(forcnn).flatten().reshape(self.k*self.w,1))
           
        X_preprocessed=np.array(train)
        if y is not None:
            y_preprocessed=[y[i] for i in range(n)]
        
            return X_preprocessed,y_preprocessed
        else :
            return X_preprocessed
   
    def fit(self,X,y=None):
        """
        X is a list of networkx graphs
        y is a list of class 
        """
        X_preprocessed,y_preprocessed=self.process_data(X,y)
        self.model.fit(X_preprocessed,y_preprocessed)
        
    def predict(self,X):
        X_preprocessed=self.process_data(X)
        return self.model.predict(X_preprocessed)



class ReceptiveFieldMaker():
    """
    nx_graph: a networkx graph. A receptive field is created over this graph
    w : the length of the node sequence 
    k : the size of the receptive field
    labeling_procedure_name : name of the labeling procedure, just implemented betweeness
    """
    def __init__(self,nx_graph,w,s=1,k=10,labeling_procedure_name='betweeness'):
        self.nx_graph=nx_graph
        self.w=w
        self.s=s
        self.k=k
        self.labeling_procedure_name=labeling_procedure_name
        self.dict_first_labeling=self.labeling_procedure(self.nx_graph)
        self.original_labeled_graph=self.dict_first_labeling['labeled_graph']

    def make_(self):
        "Result on one (w,k,length_attri) list (usually (w,k,1)) for 1D CNN "
        forcnn=[]
        f=self.select_node_sequence()
        for graph in f :
            frelabel=nx.relabel_nodes(graph,nx.get_node_attributes(graph,'labeling'))
            forcnn.append([x[1] for x in sorted(nx.get_node_attributes(frelabel,'attr_name').items(),key=lambda x:x[0])])
        return forcnn

    def betweenness_centrality_labeling(self,graph):
        result={}
        labeled_graph=nx.Graph(graph)
        centrality=list(nx.betweenness_centrality(graph).items())
        sorted_centrality=sorted(centrality,key=lambda n:n[1],reverse=True)
        dict_={}
        label=0
        for t in sorted_centrality:
            dict_[t[0]]=label
            label+=1
        nx.set_node_attributes(labeled_graph,dict_,'labeling')
        ordered_nodes=list(zip(*sorted_centrality))[0]
        
        result['labeled_graph']=labeled_graph
        result['sorted_centrality']=sorted_centrality
        result['ordered_nodes']=ordered_nodes
        return result

    def wl_normalization(self,graph):

        result={}

        labeled_graph=nx.Graph(graph)

        relabel_dict_={}
        graph_node_list=list(graph.nodes())
        for i in range(len(graph_node_list)):
            relabel_dict_[graph_node_list[i]]=i
            i+=1

        inv_relabel_dict_={v:k for k,v in relabel_dict_.items()}

        graph_relabel=nx.relabel_nodes(graph,relabel_dict_)

        label_lookup = {}
        label_counter = 0

        l_aux = list(nx.get_node_attributes(graph_relabel,'attr_name').values())
        labels = np.zeros(len(l_aux), dtype=np.int32)
        adjency_list = list([list(x[1].keys()) for x in graph_relabel.adjacency()]) #adjency list à l'ancienne comme version 1.0 de networkx


        for j in range(len(l_aux)):
            if not (l_aux[j] in label_lookup):
                label_lookup[l_aux[j]] = label_counter
                labels[j] = label_counter
                label_counter += 1
            else:
                labels[j] = label_lookup[l_aux[j]]
            # labels are associated to a natural number
            # starting with 0.

        new_labels = copy.deepcopy(labels)

        # create an empty lookup table
        label_lookup = {}
        label_counter = 0

        for v in range(len(adjency_list)):
            # form a multiset label of the node v of the i'th graph
            # and convert it to a string

            long_label = np.concatenate((np.array([labels[v]]),np.sort(labels[adjency_list[v]])))
            long_label_string = str(long_label)
            # if the multiset label has not yet occurred, add it to the
            # lookup table and assign a number to it
            if not (long_label_string in label_lookup):
                label_lookup[long_label_string] = label_counter
                new_labels[v] = label_counter
                label_counter += 1
            else:
                new_labels[v] = label_lookup[long_label_string]
        # fill the column for i'th graph in phi
        labels = copy.deepcopy(new_labels)

        dict_={inv_relabel_dict_[i]:labels[i] for i in range(len(labels))}

        nx.set_node_attributes(labeled_graph,dict_,'labeling')

        result['labeled_graph']=labeled_graph
        result['ordered_nodes']=[x[0] for x in sorted(dict_.items(), key=lambda x:x[1])]

        return result

    def labeling_procedure(self,graph):
        if self.labeling_procedure_name=='betweeness':
            return self.betweenness_centrality_labeling(graph)

    def select_node_sequence(self):
        Vsort=self.dict_first_labeling['ordered_nodes']
        f=[]
        i=0
        j=1
        while j<=self.w :

            if i<len(Vsort):
                f.append(self.receptiveField(Vsort[i]))
            else:
                f.append(self.zeroReceptiveField())
            i+=self.s
            j+=1

        return f 

    def zeroReceptiveField(self):
        graph=nx.star_graph(self.k-1) #random graph peu importe sa tete
        nx.set_node_attributes(graph,0,'attr_name')
        nx.set_node_attributes(graph,{k:k for k,v in dict(graph.nodes()).items()},'labeling')

        return graph

    def receptiveField(self,vertex):
        subgraph=self.neighborhood_assembly(vertex)
        normalized_subgraph=self.normalize_graph(subgraph,vertex)

        return normalized_subgraph


    def neighborhood_assembly(self,vertex):
        "Output networkx subgraph"
        N={vertex}
        L={vertex}
        while len(N)<self.k and len(L)>0:
            tmp=set()
            for v in L:
                tmp=tmp.union(set(self.nx_graph.neighbors(v)))
            L=tmp-N
            N=N.union(L)
        return self.nx_graph.subgraph(list(N))


    def rank_label_wrt_dict(self,subgraph,label_dict,dict_to_respect):

        all_distinc_labels=list(set(label_dict.values()))
        new_ordered_dict=label_dict     

        latest_biggest_label=0

        for label in all_distinc_labels:

            nodes_with_this_label = [x for x,y in subgraph.nodes(data=True) if y['labeling']==label]

            if len(nodes_with_this_label)>=2:

                inside_ordering=sorted(nodes_with_this_label, key=dict_to_respect.get)
                inside_order_dict=dict(zip(inside_ordering,range(len(inside_ordering))))

                for k,v in inside_order_dict.items():

                    new_ordered_dict[k]=latest_biggest_label+1+inside_order_dict[k]

                latest_biggest_label=latest_biggest_label+len(nodes_with_this_label)

            else :
                new_ordered_dict[nodes_with_this_label[0]]=latest_biggest_label+1 
                latest_biggest_label=latest_biggest_label+1

        return new_ordered_dict

    def compute_subgraph_ranking(self,subgraph,vertex,original_order_to_respect):

        labeled_graph=nx.Graph(subgraph)
        ordered_subgraph_from_centrality=self.labeling_to_root(subgraph,vertex)

        all_labels_in_subgraph_dict=nx.get_node_attributes(ordered_subgraph_from_centrality,'labeling')

        new_ordered_dict=self.rank_label_wrt_dict(ordered_subgraph_from_centrality,all_labels_in_subgraph_dict,original_order_to_respect)

        nx.set_node_attributes(labeled_graph,new_ordered_dict,'labeling') 

        return labeled_graph

    def canonicalizes(self,subgraph):

        #wl_subgraph_normalized=self.wl_normalization(subgraph)['labeled_graph']
        #g_relabel=convert_node_labels_to_integers(wl_subgraph_normalized)

        g_relabel=convert_node_labels_to_integers(subgraph)
        labeled_graph=nx.Graph(g_relabel)

        nauty_graph=NautyGraph(len(g_relabel.nodes()),directed=False)
        nauty_graph.set_adjacency_dict({n:list(nbrdict) for n,nbrdict in g_relabel.adjacency()})

        labels_dict=nx.get_node_attributes(g_relabel,'labeling')
        canonical_labeling_dict={k:canonical_labeling(nauty_graph)[k] for k in range(len(g_relabel.nodes()))}

        new_ordered_dict=self.rank_label_wrt_dict(g_relabel,labels_dict,canonical_labeling_dict)

        nx.set_node_attributes(labeled_graph,new_ordered_dict,'labeling') 

        return labeled_graph

    def normalize_graph(self,subgraph,vertex):

        "U set of vertices. Return le receptive field du vertex (un graph normalisé)"
        ranked_subgraph_by_labeling_procedure=self.labeling_procedure(subgraph)['labeled_graph']
        original_order_to_respect=nx.get_node_attributes(ranked_subgraph_by_labeling_procedure,'labeling') # à changer je pense
        subgraph_U=self.compute_subgraph_ranking(subgraph,vertex,original_order_to_respect) #ordonne les noeuds w.r.t labeling procedure

        if len(subgraph_U.nodes())>self.k:

            d=dict(nx.get_node_attributes(subgraph_U,'labeling'))    
            k_first_nodes=sorted(d,key=d.get)[0:self.k]
            subgraph_N=subgraph_U.subgraph(k_first_nodes)  

            ranked_subgraph_by_labeling_procedure=self.labeling_procedure(subgraph)['labeled_graph']
            original_order_to_respect=nx.get_node_attributes(ranked_subgraph_by_labeling_procedure,'labeling')        
            subgraph_ranked_N=self.compute_subgraph_ranking(subgraph_N,vertex,original_order_to_respect)

        elif len(subgraph_U.nodes())<self.k:
            print('shit could happen') #TODO : ajouter des dummy nodes
            N=U
            N.append(dummy_nodes(k-len(U)))
        else :
            subgraph_ranked_N=subgraph_U

        return self.canonicalizes(subgraph_ranked_N)

    def neighborhood_assembly_list(self,sequence,graph,k):
        graph_list=[]
        for v in sequence:
            nodes_of_interest=neighborhood_assembly(graph,v,k)
            graph_list.append(graph.subgraph(nodes_of_interest))
        return graph_list

    def labeling_to_root(self,graph,vertex):
        labeled_graph=nx.Graph(graph)
        source_path_lengths = nx.single_source_dijkstra_path_length(graph, vertex)
        nx.set_node_attributes(labeled_graph,source_path_lengths,'labeling')
        
        return labeled_graph











