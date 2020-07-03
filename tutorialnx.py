#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:20:05 2019

@author: amaity
"""
import os
import pprint
import numpy as np
import networkx as nx
import networkx.drawing.nx_agraph as nxdr
import networkx.algorithms as nalgs
import networkx.algorithms.dag as nxdag

def main():
    G = nx.gnp_random_graph(4,0.5,directed=True)
    H = nx.DiGraph([(u,v,{'weight':np.random.randint(0,5)}) for (u,v) in G.edges() if u<v])
    if nx.is_directed_acyclic_graph(H):
        nxdr.write_dot(H, 'fileD.dot')
        os.system('neato -Tpdf fileD.dot >fileD.pdf')
    nxdr.write_dot(G, 'fileG.dot')
    os.system('neato -Tpdf fileG.dot >fileG.pdf')

def ACTest():
    G  = nxdr.read_dot(f'dag3/random8.dot')
    TC = nx.transitive_closure(G)       # Complexity of TC
    nxdr.write_dot(TC,f'dump.dot')
    antichains_stacks = [([], list(reversed(list(nx.topological_sort(G)))))]
    pprint.pprint(antichains_stacks)
    i = 0
    while antichains_stacks:
        (antichain, stack) = antichains_stacks.pop()
        print(f'i({i}):AC:{antichain},stack:{stack}')
        #yield antichain
        j = 0
        while stack:
            x = stack.pop()
            print(f'j({j}),x:{x},stack:{stack},TC[{x}]:{TC[x]}')
            new_antichain = antichain + [x]
            new_stack = [t for t in stack if not ((t in TC[x]) or (x in TC[t]))]
            antichains_stacks.append((new_antichain, new_stack))
            j = j+1
        i = i+1
        
if __name__=="__main__":
    # main()
    ACTest()