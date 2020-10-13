import random
import os
import pprint
import utils as ut
import pandas as pd
import DnC as dnc
import PkMin as pkp
import itertools as it

def motivationalExperiment():
    Gfl = f'benchmarks/dag4/random100_2.dot' # Use f'benchmarks/motivation.dot' for motiv
    D   = 1700 # Use 425 for motiv
    p2,f2,r2,oldGfl = pkp.PkMin(Gfl,D,True)

def main():
    optimized=False
    dumDir=f'JLPEA2020ResultsRev2'
    os.system(f'mkdir -p {dumDir}')

    # Variable task graph experiments - Increasing task graph sizes
    # rLL = []
    # for N in range(4,101):
    #     D   = (17)*N
    #     Gfl = f'benchmarks/dag3/random{N}_0.dot'
    #     p1,f1,allM1,e1,r1 = pkp.PkMin(Gfl,D,False,optimized)
    #     p2,f2,allM2,e2,r2 = pkp.PkMin(Gfl,D,True,optimized)
    #     p3,f3,allM3,e3,r3 = dnc.DnCLike(Gfl,D)
    #     rLL.append({
    #         'ATGFileName'                : Gfl,
    #         'VertexSize'                 : N,
    #         'Deadline'                   : D,
    #         'pkp_DnC'                    : p3,
    #         'pkp_PkMinRoundUp'           : p2,
    #         'pkp_PkMinRoundDown'         : p1,
    #         'energy_DnC'                 : e3,
    #         'energy_PkMinRoundUp'        : e2,
    #         'energy_PkMinRoundDown'      : e1,
    #         'ATGExecTime_DnC'            : f3,
    #         'ATGExecTime_PkMinRoundUp'   : f2,
    #         'ATGExecTime_PkMinRoundDown' : f1,
    #         'runTime_DnC'                : r3,
    #         'runTime_PkMinRoundUp'       : r2,
    #         'runTime_PkMinRoundDown'     : r1
    #     })
    #     print(f'1. Processing {Gfl}')
    #     if allM1 > ut.MAXCPU or allM2 > ut.MAXCPU or allM3 > ut.MAXCPU:
    #         raise ValueError(f'1. Resource constraints violated...')
    # strName = f'randomTGWithTightDeadlineVariableSize'
    # print(f'Completed {strName} experiments')
    # dfX = pd.DataFrame(rLL)
    # dfX.to_csv(f'{dumDir}/{strName}.csv')
    

    # Variable task graph experiments - Constant task graph sizes
    rLL = []
    for id in range(68,69): # 101
        N = 100
        D   = 17*N
        Gfl = f'benchmarks/dag4/random{N}_{id}.dot'
        p1,f1,allM1,e1,r1 = pkp.PkMin(Gfl,D,False,optimized)
        os.system(f'mv ExectionTrace.csv {dumDir}/ExectionTrace{N}_{id}_PKPMinRoundDown.csv')
        p2,f2,allM2,e2,r2 = pkp.PkMin(Gfl,D,True,optimized)
        os.system(f'mv ExectionTrace.csv {dumDir}/ExectionTrace{N}_{id}_PKPMinRoundUp.csv')
        p3,f3,allM3,e3,r3 = dnc.DnCLike(Gfl,D)
        os.system(f'mv ExectionTrace.csv {dumDir}/ExectionTrace{N}_{id}_DnCLike.csv')
        rLL.append({
            'ATGFileName'                : Gfl,
            'VertexSize'                 : N,
            'Deadline'                   : D,
            'pkp_DnC'                    : p3,
            'pkp_PkMinRoundUp'           : p2,
            'pkp_PkMinRoundDown'         : p1,
            'energy_DnC'                 : e3,
            'energy_PkMinRoundUp'        : e2,
            'energy_PkMinRoundDown'      : e1,
            'ATGExecTime_DnC'            : f3,
            'ATGExecTime_PkMinRoundUp'   : f2,
            'ATGExecTime_PkMinRoundDown' : f1,
            'runTime_DnC'                : r3,
            'runTime_PkMinRoundUp'       : r2,
            'runTime_PkMinRoundDown'     : r1
        })
        print(f'2. Processing {Gfl}')
        if allM1 > ut.MAXCPU or allM2 > ut.MAXCPU or allM3 > ut.MAXCPU:
            raise ValueError(f'1. Resource constraints violated...')
    strName = f'randomTGWithTightDeadlineConstantSize'
    print(f'Completed {strName} experiments')
    dfX = pd.DataFrame(rLL)
    dfX.to_csv(f'{dumDir}/{strName}.csv')
    
    # Variable (monotonically increasing) deadline experiments
    # rLL = []
    # for fac in range(17,118): # 118
    #     N = 100
    #     D = fac*N
    #     Gfl = f'benchmarks/dag3/random{N}_0.dot'
    #     p1,f1,allM1,e1,r1 = pkp.PkMin(Gfl,D,False,optimized)
    #     p2,f2,allM2,e2,r2 = pkp.PkMin(Gfl,D,True,optimized)
    #     p3,f3,allM3,e3,r3 = dnc.DnCLike(Gfl,D)
    #     rLL.append({
    #         'ATGFileName'                : Gfl,
    #         'VertexSize'                 : N,
    #         'Deadline'                   : D,
    #         'pkp_DnC'                    : p3,
    #         'pkp_PkMinRoundUp'           : p2,
    #         'pkp_PkMinRoundDown'         : p1,
    #         'energy_DnC'                 : e3,
    #         'energy_PkMinRoundUp'        : e2,
    #         'energy_PkMinRoundDown'      : e1,
    #         'ATGExecTime_DnC'            : f3,
    #         'ATGExecTime_PkMinRoundUp'   : f2,
    #         'ATGExecTime_PkMinRoundDown' : f1,
    #         'runTime_DnC'                : r3,
    #         'runTime_PkMinRoundUp'       : r2,
    #         'runTime_PkMinRoundDown'     : r1
    #     })
    #     print(f'3. Processing {Gfl}')
    #     if allM1 > ut.MAXCPU or allM2 > ut.MAXCPU or allM3 > ut.MAXCPU:
    #         raise ValueError(f'1. Resource constraints violated...')
    # strName = f'randomTGWithRelaxedDeadlineConstantSize'
    # print(f'Completed {strName} experiments')
    # dfX = pd.DataFrame(rLL)
    # dfX.to_csv(f'{dumDir}/{strName}.csv')
    


if __name__=="__main__":
    main()
    # motivationalExperiment()
    # Gfl = f'dag3/random82_0.dot'
    # G = ut.ATG(Gfl)
    # print(G.getMinAlloc(2,32))
