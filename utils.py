import logging
import numpy as np
import argparse
import matplotlib.colors as mcol
from matplotlib import cm
import networkx as nx 
from custom_errors import NotRepresentableError

def indices_to_one_hot(number, nb_classes,label_dummy=-1):
    """Convert an iterable of indices to one-hot encoded labels."""
    
    if number==label_dummy:
        return np.zeros(nb_classes)
    else:
        return np.eye(nb_classes)[number]

def graph_colors(nx_graph,vmin=0,vmax=9):
    #cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["blue","red"])
    #cm1 = mcol.Colormap('viridis')

    cnorm = mcol.Normalize(vmin=vmin,vmax=vmax)
    cpick = cm.ScalarMappable(norm=cnorm,cmap='Set1')
    cpick.set_array([])
    val_map = {}
    for k,v in nx.get_node_attributes(nx_graph,'attr_name').items():
        if isinstance(v,int):
            val_map[k]=cpick.to_rgba(v)
        elif isinstance(v,list):
            if len(v)>1:
                raise NotRepresentableError('Feature must be one dimensionnal in order to be displayed')
            val_map[k]=cpick.to_rgba(v[0])
            
    colors=[]
    for node in nx_graph.nodes():
        colors.append(val_map[node])
    return colors
    
def pos_diff(pos,x_off=0,y_off=0):
    pos_higher = {}
    for k, v in pos.items():
        pos_higher[k] = (v[0]+x_off, v[1]+y_off) 
    return pos_higher
    
def allnan(v): #fonctionne juste pour les dict de tuples
    from math import isnan
    import numpy as np
    return np.all(np.array([isnan(k) for k in list(v)]))
    
def dict_argmax(d):
    l={k:v for k, v in d.items() if not allnan(v)}
    return max(l,key=l.get)
def dict_argmin(d):
    return min(d, key=d.get)

def read_files(mypath):
    from os import listdir
    from os.path import isfile, join

    return [f for f in listdir(mypath) if isfile(join(mypath, f))]

def per_section(it, is_delimiter=lambda x: x.isspace()):
    ret = []
    for line in it:
        if is_delimiter(line):
            if ret:
                yield ret  # OR  ''.join(ret)
                ret = []
        else:
            ret.append(line.rstrip())  # OR  ret.append(line)
    if ret:
        yield ret

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

