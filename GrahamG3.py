import random
import pprint
import math
import utils as ut
import networkx as nx
import networkx.drawing.nx_agraph as nxdr
import matplotlib.pyplot as plt
import numpy as np
import heapq
import time
import copy

def GrahamPkp(T,M,atg):
    """
        Same as G3 introduced mentioned in
        "A Divide and Conquer Algorithm for DAG
        Scheduling under Power Constraints"

        For all the ready tasks
        1. Compute the slowest execution time of all the tasks
        2. 
    """
    freeM  = M      # Free cores
    IS     = []                       # Data structure to hold schedule
    exT    = []                       # Priority queue of executing nodes
    U      = set()                    # Set of already finished nodes
    T2     = set(copy.deepcopy(T))    # Local copy of nodes to be processed
    readyT = atg.getReadyNodes2(T2,U)  # Set of enabled nodes which are not already executed
    time   = 0
    pass