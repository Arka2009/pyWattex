import random
import pprint
import utils as ut
import networkx as nx
import networkx.drawing.nx_agraph as nxdr
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
import DnC as dnc
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import PkMin as pkp
import os

def motivationalExperiment():
    Gfl = f'benchmarks/dag4/random100_2.dot' # Use f'benchmarks/motivation.dot' for motiv
    D   = 1700 # Use 425 for motiv
    p2,f2,r2,oldGfl = pkp.PkMin(Gfl,D,True)

def main():
    optimized=False
    os.system(f'mkdir -p Results')
    flRandomTGConstantTaskGraphSize = open(f'Results/randomTGWithConstantSize.csv','w')
    flVariableDeadline = open(f'Results/randomDeadlines.csv','w')
    flRunTimeAndVariableTaskGraphSize = open(f'Results/randomTGWithVariableSize.csv','w')
    print(f'WorkloadId,NormalizedPerfPkp,NormalizedPerfEnergy',file=flRandomTGConstantTaskGraphSize)
    print(f'WorkloadId,NormalizedPerfPkp,NormalizedPerfEnergy',file=flVariableDeadline)
    print(f'NumTasks,NormalizedPerf,NormalizedPerfEnergy,DnCRuntime,PkMinRuntime',file=flRunTimeAndVariableTaskGraphSize)

    # # Variable task graph experiments - Increasing task graph sizes
    for N in range(4,101):
        D   = (17)*N
        Gfl = f'benchmarks/dag3/random{N}_0.dot'
        p1,f1,allM1,e1,r1 = dnc.DnCLike(Gfl,D)
        p2,f2,allM2,e2,r2 = pkp.PkMin(Gfl,D,optimized)
        print(f'{N},{p2/p1},{e2/e1},{r1},{r2}',file=flRunTimeAndVariableTaskGraphSize)
        # print(f'{N},{p2/p1},{e2/e1},{r1},{r2}')
        print(f'1. Processing {Gfl}')
        if allM1 > ut.MAXCPU and allM2 > ut.MAXCPU :
            raise ValueError(f'1 Resource constraints violated...')
    print(f'Completed Vraible TG Variable Size experiments')
    

    # Variable task graph experiments - Constant task graph sizes
    for id in range(1,101): # 101
        N = 100
        D   = 17*N
        Gfl = f'benchmarks/dag4/random{N}_{id}.dot'
        p1,f1,allM1,e1,r1 = dnc.DnCLike(Gfl,D)
        os.system(f'mv ExectionTrace.csv Results/ExectionTrace{N}_{id}_DnCLike.csv')
        p2,f2,allM2,e2,r2 = pkp.PkMin(Gfl,D,optimized)
        os.system(f'mv ExectionTrace.csv Results/ExectionTrace{N}_{id}_PKPMin.csv')
        print(f'{id},{p2/p1},{e2/e1}',file=flRandomTGConstantTaskGraphSize)
        print(f'{id},{p2/p1:.2f},{e2/e1:.2f},{allM1},{allM2},{f1/D:.2f},{f2/D:.2f}')
        print(f'2. Processing {Gfl}')
        if allM1 > ut.MAXCPU and allM2 > ut.MAXCPU :
            raise ValueError(f'2 Resource constraints violated...')
    print(f'Completed numRandom experiments')
    
    
    # Variable (monotonically increasing) deadline experiments
    for fac in range(17,118): # 118
        N = 100
        D = fac*N
        Gfl = f'benchmarks/dag3/random{N}_0.dot'
        p1,f1,allM1,e1,r1 = dnc.DnCLike(Gfl,D)
        p2,f2,allM2,e2,r2 = pkp.PkMin(Gfl,D,optimized)
        print(f'{D},{p2/p1},{e2/e1}',file=flVariableDeadline)
        # print(f'{D},{p2/p1:.2f},{e2/e1:.2f},{allM1},{allM2},{f1/D:.2f},{f2/D:.2f}')
        print(f'3. Processing {Gfl}')
        if allM1 > ut.MAXCPU and allM2 > ut.MAXCPU :
            raise ValueError(f'3 Resource constraints violated...')
    print(f'Completed numDeadlines experiments')
    

    flRandomTGConstantTaskGraphSize.close()
    flVariableDeadline.close()
    flRunTimeAndVariableTaskGraphSize.close()

if __name__=="__main__":
    main()
    # motivationalExperiment()
    # Gfl = f'dag3/random82_0.dot'
    # G = ut.ATG(Gfl)
    # print(G.getMinAlloc(2,32))
