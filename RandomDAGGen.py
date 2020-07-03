import random
import pprint
import utils as ut
import networkx as nx
import networkx.drawing.nx_agraph as nxdr
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
from scipy.stats import bernoulli

def DAGGen(path,N,p,id):
    """
        Erdos-Renyi Random digraph (sort of)
        with number of Nodes = N and edge
        connection probability = p
    """
    G = nx.DiGraph()
    for u in range(N):
        b=random.choice(ut.BENCH)
        G.add_node(u,bench=b,aet=ut.AETDICT[b],bet=ut.BETDICT[b],ap=ut.APDICT[b],bp=ut.BPDICT[b],stack=1,alloc=0)
    
    rv = bernoulli(p)
   
    for u in range(N):
        for v in range(u+1,N):
            if rv.rvs() == 1 :
                G.add_edge(u,v)

    if nx.is_directed_acyclic_graph(G):
        print(f'The Digraph{N}_{id} is acyclic')
    
    nxdr.write_dot(G,f'{path}/random{N}_{id}.dot')


if __name__=="__main__":
    for N in range(5,101):
        DAGGen(f'/home/amaity/Dropbox/NUS-Research/HardPkpMin3/dag3',N,0.68,0)
    
    N = 100
    for id in range(1,101):
        DAGGen(f'/home/amaity/Dropbox/NUS-Research/HardPkpMin3/dag4',N,0.68,id)