
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
import testcvxopt as tcvx

def cvxallocTest():
    """
        Perform an allocation
        for the topologically
        sorted set of nodes in T
    """
    N  = 4
    D  = 2
    a1 = [0.9306707270772263, 0.781253569721745, 1.96668139375962, 1.7920940392776321]
    b1 = [1.3267719165086922, 0.765264919613304, 1.3159522447618568, 0.5911815286630802]
    a2 = [1.5736020733308043, 0.5113181889897235, 0.30285882145827975, 0.4394272763462338]
    b2 = [0.43724930121492356, 1.4835425054235123, 1.1986166932894586, 0.862884848808483]

    lb  = [1 for _ in range(N)] + [0.0]
    ub  = [ut.MAXCPU for _ in range(N)]+[float('inf')]

    opt = tcvx.CPPCVXOptimizer()
    x   = [1 for _ in range(N)]+[0.0]
    opt.setParams(N,x,a1,b1,a2,b2,1,ut.MAXCPU)
    opt.optimize(D)
    xopt = opt.getOpt()
    print(xopt[:-1])


def PkMin(fl2,D,debugPrint=False):
    pkp    = 0.0 
    atg = ut.ATG(fl2)
    tol = 1e-8
    s1  = time.time()
    i = 0
    pkp,finish,_ = atg.initalloc(D)
    completelySerialized = False
    
    while (not completelySerialized):
        if i == 0 :
            bestAtg     = copy.deepcopy(atg)
            bestPkp     = pkp
            bestFin     = finish
        
        if debugPrint :
            pkp2,finish2,_ = atg.getTotalEtPower()
            print(f'iter_Beg@{i}|Et:{finish2},Pkp:{pkp2},ATG:{atg}\n')

        # CVX Opt Step
        pkp_cvx,finish_cvx,_ = atg.cvxalloc(D)
        if (pkp_cvx < bestPkp) and (finish_cvx <= D) : # Save the best encountered allocation
            bestAtg  = copy.deepcopy(atg)
            bestPkp  = pkp_cvx
            bestFin  = finish_cvx
        
        if debugPrint :
            pkp2,finish2,_ = atg.getTotalEtPower()
            print(f'iter_AftCVXOpt@{i}|Et:{finish2},Pkp:{pkp2},ATG:{atg}\n')

        # DAG merging
        ac = atg.computeBestPair(ut.MAXCPU)
        if ac:
            a,c = ac
            atg.mergeNodes(a,c)
        else :
            completelySerialized = True
        i += 1

    s2  = time.time()
    bestAtg.setScheduled(True)
    verpkp,verfinish,maxM,energy = bestAtg.getTotalEtPower()
    return (verpkp,verfinish,maxM,energy,(s2-s1))
