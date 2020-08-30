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

def motivationalExperiment():
    Gfl = f'benchmarks/dag4/random100_2.dot' # Use f'benchmarks/motivation.dot' for motiv
    D   = 1700 # Use 425 for motiv
    p2,f2,r2,oldGfl = ut.PkMin(Gfl,D,True)

def main():
    fl1 = open(f'numTasks.csv','w')
    fl2 = open(f'numRandom.csv','w')
    fl3 = open(f'numDeadlines.csv','w')
    fl4 = open(f'runtime.csv','w')
    print(f'NumTasks,NormalizedPerf',file=fl1)
    print(f'WorkloadId,NormalizedPerf',file=fl2)
    print(f'Deadlines,NormalizedPerf',file=fl3)
    print(f'NumTasks,DnCRuntime,PkMinRuntime',file=fl4)
    # flT = open('motiv.txt','w')

    # for N in range(50,51):
        # D   = (17)*N
        # Gfl = f'benchmarks/dag3/random{N}_0.dot'
        # print(f'Minimum for {}')
        # p2,f2,r2,oldGfl = ut.PkMin(Gfl,D)
        # p1,mMax,f1,r1 = dnc.DnCLike(Gfl,D)
        # print(f'DnCLike({N})|Pkp:{p1} with {mMax} cores, Et:{f1},Deadline:{D}')
        # print(f'PkMin({N})|Pkp:{p2},Et:{f2},Deadline:{D}\n')
        # print(oldGfl,file=flT)
        # print(f'r1:{r1},r2:{r2}')
        # print(f'{N},{p2/p1}',file=fl1)
        # print(f'{N},{r1},{r2}',file=fl4)
    # print(f'Completed numTasks experiments')
    # flT.close()

    for id in range(2,20): # 101
        N = 100
        D   = 17*N
        Gfl = f'benchmarks/dag4/random{N}_{id}.dot'
        p2,f2,r2,_= ut.PkMin(Gfl,D)
        p1,f1,r1,_ = dnc.DnCLike(Gfl,D)
        print(f'{id},{p2/p1}',file=fl2)
        print(f'{id},{p2/p1}')
        # pprint.pprint({
        #     'id' : id,
        #     'pkp_DnCLike':p1,
        #     'pkp_PkMin':p2,
        #     'Et_DnCLike':f1/D,
        #     'Et_PkMin':f2/D
        # })
        print(f'\n')
    print(f'Completed numRandom experiments')
    
    # p2=np.Infinity
    # for fac in range(17,118): # 118
    #     N = 100
    #     D = fac*N
    #     Gfl = f'benchmarks/dag3/random{N}_0.dot'
    #     p2,f2,r2,oldGfl = ut.PkMin(Gfl,D)
    #     p1,mMax,f1,r = dnc.DnCLike(Gfl,D)
    #     print(f'{D},{p2/p1}',file=fl3)
    #     print(f'{D},{p2/p1}')
    # print(f'Completed numDeadlines experiments')

    # fl1.close()
    fl2.close()
    fl3.close()
    # fl4.close()

if __name__=="__main__":
    main()
    # motivationalExperiment()
    # Gfl = f'dag3/random82_0.dot'
    # G = ut.ATG(Gfl)
    # print(G.getMinAlloc(2,32))