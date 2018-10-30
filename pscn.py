"Patchy San"

from networkx import nx
from networkx import convert_node_labels_to_integers
from pynauty.graph import canonical_labeling,Graph
import copy
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Model,Sequential
from keras.layers import Conv1D,Dense,Dropout,Flatten
import numpy as np
import time
import tensorflow as tf
import utils


class PSCN():
    def __init__(self,w,s=1,k=10
                 ,labeling_procedure_name='betweeness'
                 ,epochs=150,batch_size=25
                 ,verbose=0
                 ,use_node_deg=False
                 ,use_preprocess_data=False
                 ,gpu=False
                 ,multiclass=None
                 ,one_hot=0
                 ,attr_dim=1
                 ,dummy_value=-1):
        """
        w : width parameter
        s: stride parameter
        k: receptive field size paremeter
        labeling_procedure_name : the labeling procedure for ranking the nodes between them. Only betweeness centrality is implemented.
        epochs: number of epochs for the CNN
        batch_size : batch size for training the CNN
        use_node_deg : wether to use node degree as label for unlabeled graphs (IMDB for e.g)
        multiclass : if the classification is not binary it is the number of classes
        one_hot : if nodes attributes are discrete it is the number of unique attributes
        attr_dim : if nodes attributes are multidimensionnal it is the dimension of the attributes
        dummy_value  which value should be used for dummy nodes (see paper)
        """
        self.w=w
        self.s=s
        self.k=k
        self.labeling_procedure_name=labeling_procedure_name
        self.epochs=epochs
        self.use_node_deg=use_node_deg
        self.batch_size=batch_size
        self.verbose=verbose
        self.use_preprocess_data=use_preprocess_data
        self.gpu=gpu
        self.multiclass=multiclass
        self.one_hot=one_hot
        self.attr_dim=attr_dim
        self.dummy_value=dummy_value
        self.model = KerasClassifier(build_fn=self.create_model
                                     ,epochs=self.epochs, 
                                     batch_size=self.batch_size, verbose=self.verbose)
        self.times_process_details={}
        self.times_process_details['normalized_subgraph']=[]
        self.times_process_details['neigh_assembly']=[]
        self.times_process_details['canonicalizes']=[]
        self.times_process_details['compute_subgraph_ranking']=[]
        self.times_process_details['labeling_procedure']=[]       
        self.times_process_details['first_labeling_procedure']=[]
        
        if self.one_hot>0:
            self.attr_dim=self.one_hot

        
    def create_model(self):
       model=Sequential()
       model.add(Conv1D(filters=16,kernel_size=self.k,strides=self.k,input_shape=(self.w*self.k,self.attr_dim)))    
       model.add(Conv1D(filters=8,kernel_size=10,strides=1))
       model.add(Flatten())
       model.add(Dense(128,activation="relu",name='embedding_layer'))
       model.add(Dropout(0.5))
       if self.multiclass is not None :
           model.add(Dense(self.multiclass, activation='softmax'))
           model.compile(loss="categorical_crossentropy",optimizer="rmsprop",metrics=["accuracy"])
       else:
           model.add(Dense(1,activation="sigmoid"))
           model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=["accuracy"])    
       return model
   
    def process_data(self,X,y=None): # X is a list of Graph objects
        start=time.time()
        n=len(X)
        train=[]
        for i in range(n):
            rfMaker=ReceptiveFieldMaker(X[i].nx_graph,w=self.w,k=self.k,s=self.s
                                        ,labeling_procedure_name=self.labeling_procedure_name
                                        ,use_node_deg=self.use_node_deg,one_hot=self.one_hot,dummy_value=self.dummy_value)
            forcnn=rfMaker.make_()
            self.times_process_details['neigh_assembly'].append(np.sum(rfMaker.all_times['neigh_assembly']))
            self.times_process_details['normalized_subgraph'].append(np.sum(rfMaker.all_times['normalized_subgraph']))
            self.times_process_details['canonicalizes'].append(np.sum(rfMaker.all_times['canonicalizes']))
            self.times_process_details['compute_subgraph_ranking'].append(np.sum(rfMaker.all_times['compute_subgraph_ranking']))
            self.times_process_details['labeling_procedure'].append(np.sum(rfMaker.all_times['labeling_procedure']))
            self.times_process_details['first_labeling_procedure'].append(np.sum(rfMaker.all_times['first_labeling_procedure']))
            
            train.append(np.array(forcnn).flatten().reshape(self.k*self.w,self.attr_dim))

        X_preprocessed=np.array(train)
        end=time.time()
        print('Time preprocess data in s',end-start) 
        if y is not None:
            y_preprocessed=[y[i] for i in range(n)]
            return X_preprocessed,y_preprocessed
        else :
            return X_preprocessed
   
    def fit(self,X,y=None): 
        if not self.use_preprocess_data:
            X_preprocessed,y_preprocessed=self.process_data(X,y)
        else:
            X_preprocessed=X
            y_preprocessed=y
        start=time.time()
        if self.gpu:
            with tf.device('/gpu:0'):
                if self.verbose >0:
                    print('Go for GPU')
                self.model.fit(X_preprocessed,y_preprocessed)
        else:
            self.model.fit(X_preprocessed,y_preprocessed)
        end=time.time()
        print('Time fit data in s',end-start)
        
    def predict(self,X):
        if not self.use_preprocess_data:
            X_preprocessed=self.process_data(X)
        else:
            X_preprocessed=X
        return self.model.predict(X_preprocessed)
    
    def return_embedding(self,X):
        X_preprocessed=self.process_data(X)
        layer_output = Model(inputs=self.model.model.input,
                                 outputs=self.model.model.get_layer('embedding_layer').output)        
        return layer_output.predict(X_preprocessed)



class ReceptiveFieldMaker():
    def __init__(self,nx_graph,w,s=1,k=10,labeling_procedure_name='betweeness',use_node_deg=False,one_hot=False,dummy_value=-1):
        self.nx_graph=nx_graph
        self.use_node_deg=use_node_deg
        if self.use_node_deg:
            node_degree_dict=dict(self.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(self.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(self.nx_graph,normalized_node_degree_dict,'attr_name')
        self.all_times={}
        self.all_times['neigh_assembly']=[]
        self.all_times['normalized_subgraph']=[]
        self.all_times['canonicalizes']=[]   
        self.all_times['compute_subgraph_ranking']=[]
        self.all_times['labeling_procedure']=[]
        self.all_times['first_labeling_procedure']=[]
        self.w=w
        self.s=s
        self.k=k
        self.dummy_value=dummy_value
        self.exists_dummies=False
        self.one_hot=one_hot
        self.labeling_procedure_name=labeling_procedure_name
        
        if self.labeling_procedure_name=='approx_betweeness':
            st=time.time()
            self.dict_first_labeling=self.betweenness_centrality_labeling(self.nx_graph,approx=int(len(self.nx_graph.nodes())/5)+1) 
            self.labeling_procedure_name='betweeness'
            end=time.time()
            self.all_times['first_labeling_procedure'].append(end-st)
        elif self.labeling_procedure_name=='betweeness':
            st=time.time()
            self.dict_first_labeling=self.betweenness_centrality_labeling(self.nx_graph) 
            end=time.time()
            self.all_times['first_labeling_procedure'].append(end-st)   
        else :
            st=time.time()
            self.dict_first_labeling=self.labeling_procedure(self.nx_graph) 
            end=time.time()
            self.all_times['first_labeling_procedure'].append(end-st)    
            
        self.original_labeled_graph=self.dict_first_labeling['labeled_graph']



    def make_(self):
        "Result on one (w,k,length_attri) list (usually (w,k,1)) for 1D CNN "
        forcnn=[]
        self.all_subgraph=[]
        f=self.select_node_sequence()
        for graph in f :
            frelabel=nx.relabel_nodes(graph,nx.get_node_attributes(graph,'labeling')) #rename the nodes wrt the labeling
            self.all_subgraph.append(frelabel)
            if self.one_hot>0:
                forcnn.append([utils.indices_to_one_hot(x[1],self.one_hot) for x in sorted(nx.get_node_attributes(frelabel,'attr_name').items(),key=lambda x:x[0])])
            else:
                forcnn.append([x[1] for x in sorted(nx.get_node_attributes(frelabel,'attr_name').items(),key=lambda x:x[0])])
        return forcnn
    
    def labeling_procedure(self,graph):   
        st=time.time()
        if self.labeling_procedure_name=='betweeness':
            a=self.betweenness_centrality_labeling(graph)
        end=time.time()
        self.all_times['labeling_procedure'].append(end-st)
        return a
    
    def betweenness_centrality_labeling(self,graph,approx=None):
        result={}
        labeled_graph=nx.Graph(graph)
        if approx is None:
            centrality=list(nx.betweenness_centrality(graph).items())
        else:
            centrality=list(nx.betweenness_centrality(graph,k=approx).items())
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
        nx.set_node_attributes(graph,self.dummy_value,'attr_name')
        nx.set_node_attributes(graph,{k:k for k,v in dict(graph.nodes()).items()},'labeling')

        return graph

    def receptiveField(self,vertex):
        st=time.time()
        subgraph=self.neighborhood_assembly(vertex)
        ed=time.time()
        self.all_times['neigh_assembly'].append(ed-st)
        normalized_subgraph=self.normalize_graph(subgraph,vertex)
        ed2=time.time()
        self.all_times['normalized_subgraph'].append(ed2-ed)


        return normalized_subgraph


    def neighborhood_assembly(self,vertex):
        "Output a set of neighbours of the vertex"
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
        
        st=time.time()

        labeled_graph=nx.Graph(subgraph)
        ordered_subgraph_from_centrality=self.labeling_to_root(subgraph,vertex)

        all_labels_in_subgraph_dict=nx.get_node_attributes(ordered_subgraph_from_centrality,'labeling')

        new_ordered_dict=self.rank_label_wrt_dict(ordered_subgraph_from_centrality,all_labels_in_subgraph_dict,original_order_to_respect)

        nx.set_node_attributes(labeled_graph,new_ordered_dict,'labeling') 
        ed=time.time()
        self.all_times['compute_subgraph_ranking'].append(ed-st)
        return labeled_graph

    def canonicalizes(self,subgraph):
        
        st=time.time()

        #wl_subgraph_normalized=self.wl_normalization(subgraph)['labeled_graph'] 
        #g_relabel=convert_node_labels_to_integers(wl_subgraph_normalized)

        g_relabel=convert_node_labels_to_integers(subgraph)
        labeled_graph=nx.Graph(g_relabel)

        nauty_graph=Graph(len(g_relabel.nodes()),directed=False)
        nauty_graph.set_adjacency_dict({n:list(nbrdict) for n,nbrdict in g_relabel.adjacency()})

        labels_dict=nx.get_node_attributes(g_relabel,'labeling')
        canonical_labeling_dict={k:canonical_labeling(nauty_graph)[k] for k in range(len(g_relabel.nodes()))}

        new_ordered_dict=self.rank_label_wrt_dict(g_relabel,labels_dict,canonical_labeling_dict)

        nx.set_node_attributes(labeled_graph,new_ordered_dict,'labeling') 
        
        ed=time.time()
        self.all_times['canonicalizes'].append(ed-st)

        return labeled_graph

    def normalize_graph(self,subgraph,vertex):

        "U set of vertices. Return le receptive field du vertex (un graph normalisé)"
        ranked_subgraph_by_labeling_procedure=self.labeling_procedure(subgraph)['labeled_graph']
        original_order_to_respect=nx.get_node_attributes(ranked_subgraph_by_labeling_procedure,'labeling')
        subgraph_U=self.compute_subgraph_ranking(subgraph,vertex,original_order_to_respect) #ordonne les noeuds w.r.t labeling procedure

        if len(subgraph_U.nodes())>self.k:

            d=dict(nx.get_node_attributes(subgraph_U,'labeling'))    
            k_first_nodes=sorted(d,key=d.get)[0:self.k]
            subgraph_N=subgraph_U.subgraph(k_first_nodes)  

            ranked_subgraph_by_labeling_procedure=self.labeling_procedure(subgraph)['labeled_graph']
            original_order_to_respect=nx.get_node_attributes(ranked_subgraph_by_labeling_procedure,'labeling')        
            subgraph_ranked_N=self.compute_subgraph_ranking(subgraph_N,vertex,original_order_to_respect)

        elif len(subgraph_U.nodes())<self.k:
            subgraph_ranked_N=self.add_dummy_nodes_at_the_end(subgraph_U)
        else :
            subgraph_ranked_N=subgraph_U

        return self.canonicalizes(subgraph_ranked_N)

    def add_dummy_nodes_at_the_end(self,nx_graph): #why 0 ??
        self.exists_dummies=True
        g=nx.Graph(nx_graph)
        keys=[k for k,v in dict(nx_graph.nodes()).items()]
        labels=[v for k,v in dict(nx.get_node_attributes(nx_graph,'labeling')).items()]
        j=1
        while len(g.nodes())<self.k:
            g.add_node(max(keys)+j,attr_name=self.dummy_value,labeling=max(labels)+j)
            j+=1
        return g
            

    def labeling_to_root(self,graph,vertex):
        labeled_graph=nx.Graph(graph)
        source_path_lengths = nx.single_source_dijkstra_path_length(graph, vertex)
        nx.set_node_attributes(labeled_graph,source_path_lengths,'labeling')
        
        return labeled_graph











