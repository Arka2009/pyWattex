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

def DnCDAGSchedule(T,M,atg):
    """
        EXPERIMENTAL IMPLEMENTATION
        ---------------------------

        Graham's list scheduling

        Format of IS := <node-id,alloc,start,finish>
        T is just a collection of nodes
    """
    assert atg.valid, f'DAG not initialized'
    # pprint.pprint(readyT)
    readyM = M      # Free cores
    IS     = []                       # Data structure to hold schedule
    exT    = []                       # Priority queue of executing nodes
    U      = set()                    # Set of already finished nodes
    T2     = set(copy.deepcopy(T))    # Local copy of nodes to be processed
    readyT = atg.getReadyNodes2(T2,U)  # Set of enabled nodes which are not already executed
    time   = 0
    allocSuccess = True

    # print(f'TotalTask:{T2}')
    while len(T2) > 0:
        # Attempt to execute a task
        if len(readyT) > 0 :
            u = readyT.pop() # pop a single task/node
            m = atg.getPace(u,M)
            if readyM >= m : # There are enough cores and tasks to execute
                start  = time
                # print(f'M:{M},readyM:{readyM},m:{m}')
                finish = start+atg.getExecutionTime(u,m)
                heapq.heappush(exT,(finish,u,m))
                IS.append((u,m,start,finish))
                readyM -= m
                allocSuccess = True                
            else :
                allocSuccess = False
                readyT.add(u)
        else :
            allocSuccess = False
        
        # Advance the time, when not successful
        if not allocSuccess :
            if (len(exT) > 0) :
                finish,v,m2 = heapq.heappop(exT)
                executingNodes = set([u[1] for u in exT])
                U.add(v)
                T2.remove(v)
                readyM += m2
                readyT = atg.getReadyNodes2(T,U) - executingNodes
                time = finish
            else :
                print(f'Graham({time:.2f})|readyT:{readyT},readyM:{readyM},T:{T2},exT:{exT},m:{m},u:{u}')
                raise ValueError(f'Nobody is executing')
    return IS


def checkIndependence(T,atg):
    """
        T has a format
        similar to IS.
    """
    nodesT = set([u[0] for u in T])
    readyT = set(atg.getReadyNodes(nodesT,set()))
    if readyT == nodesT:
        return True
    else :
        return False

def ScheduleInd(T,M,D,atg):
    """
        Format of T same as IS
    """
    if not T:
        oldT = T
        return oldT
    
    assert checkIndependence(T,atg), f'taskset:{T} are not independent'
    tol = 1e-8
    
    # Apply NDFH iteratively
    pkp    = np.max(list(map(lambda u : atg.getPower(u[0],u[1]),T)))
    finish = np.max(list(map(lambda u : u[3],T)))
    oldT   = copy.deepcopy(T)
    oldpkp = pkp
    i = 0
    flag = False
    while True:
        # find the highest power phase
        # print(f'Iteration:{i},pkp:{pkp},finish:{finish},D:{D}')
        idx               = np.argmax(list(map(lambda u: atg.getPower(u[0],u[1]),T)))
        u,m,start,finish  = T[idx]
        m                -= 1
        # print(f'u:{u},alloc:{m}')
        finish            = start + atg.getExecutionTime(u,m)
        T[idx]            = (u,m,start,finish)

        # NDFH schedule
        T       = NDFH(T,M,D,atg)
        
        # print(f'i({i})--------------')

        if not T:
            print('TASK EMPTY')
            break

        pkp     = np.max(list(map(lambda u : atg.getPower(u[0],u[1]),T)))
        finish  = np.max(list(map(lambda u : u[3],T)))

        # Deadline Missed
        if finish > D:
            print('DEADLINE MISSED')
            # flag = True
            break

        # Covergence
        # if np.abs(oldpkp-pkp) < tol:
        #     break
        
        # Old configuration is better
        if oldpkp < pkp :
            print('OLD CONFIG BETTER')
            break
    
        # Number of cores is zero
        if m <= 0:
            # raise ValueError(f'm became ZEROM')
            break

        i += 1
        oldpkp = pkp
        oldT   = copy.deepcopy(T)
    # pprint.pprint(oldT)
    # if flag:
    #     print("Execution Time exceeds deadline")
    #     pprint.pprint(oldT)
    #     pprint.pprint(T)
    return oldT

def ScheduleInd2(T,M,D,atg):
    oldT = NDFH(T,M,D,atg)
    return oldT

def NDFH(T,M,D,atg):
    """
        Format of T is
        same as IS := <node-id,alloc,start,finish>
    """
    # T2 = copy.deepcopy(T)
    augT = [(id,m,atg.getExecutionTime(id,m),atg.getPower(id,m)) for id,m,_,_ in T]
    augT.sort(key=lambda u : u[3],reverse=True)
    level  = 0
    start  = 0.0
    finish = 0.0
    newT   = []
    totalM = 0
    # print(f'Size of NDFH:{len(augT)}')
    # pprint.pprint(augT)
    for t in augT:
        id,m,et,p = t
        if (start + et < D) : # Fits : Schedule serially (Horizontal stretch)
            finish = start + et
            newT.append((id,m,start,finish))
            start = finish
        else : 
            if totalM + m >= M : # Resource constraint violated, schedule serially
                finish = start + et
                newT.append((id,m,start,finish))
                start = finish
            else :
                start  = 0                            # Vertical stretch
                finish = start + et
                newT.append((id,m,start,finish))
                totalM += m
                level += 1
    # pprint.pprint(newT)
    return newT 


def NDFH2(T,M,D,atg):
    """
        Format of T is
        same as IS := <node-id,alloc,start,finish>.
        Find the minimum allocation for
        all tasks such that they fit in a width 
        of D and then perform NDFH strip packing.
    """
    # Compute the minimum value of cores required for each task
    # W    = atg.getWidth()
    T2   = [(u,atg.getMinAlloc(u,D,M)) for u,_,_,_ in T]
    augT = [(u,m,atg.getExecutionTime(u,m)) for u,m in T2] # Why the magic number 4 
    #etl  = lambda u2,m2 : atg.getExecutionTime(u2,m2)
    augT.sort(key=lambda u : u[2],reverse=True)
    level  = 0
    start  = 0.0
    finish = 0.0
    newT   = []
    totalM = 0
    # print(f'Size of NDFH:{len(augT)}')
    # pprint.pprint(augT)
    for t in augT:
        id,m,et = t
        if (start + et < D) : # Fits : Schedule serially (Horizontal stretch)
            finish = start + et
            newT.append((id,m,start,finish))
            start = finish
        else : 
            if totalM + m >= M : # Resource constraint violated, schedule serially
                finish = start + et
                newT.append((id,m,start,finish))
                start = finish
            else :              # Schedule parallely
                start  = 0                            # Vertical stretch
                finish = start + et
                newT.append((id,m,start,finish))
                totalM += m
                level += 1
    # pprint.pprint(newT)
    return newT 

def getStartFinish(T):
    """
        Get the start and finish time of
        a taskset
    """
    if not T:
        return (0,0)
    startT  = np.min(list(map(lambda u : u[2],T)))
    finishT = np.max(list(map(lambda u : u[3],T)))
    return (startT,finishT)

def offsetTask(T,time):
    """
        Offset the tasks
        in-place
    """
    if not T:
        return []
    newT = []
    for t in T:
        u,m,start,finish = t
        t = (u,m,start+time,finish+time)
        newT.append(t)
    return newT

def DnCRecursive(Tp,IS,M,D,atg,start,finish):
    """
        Recursive implementation.
        Format of Tp is same as IS
    """
    if len(Tp) == 0:
        return []
    
    mid  = start + (finish - start)*0.5
    Tmid = []
    Tbef = []
    Taft = []
    for t in Tp:
        if start <= t[2] <= t[3] < mid :
            Tbef.append(t)
        elif start <= t[2] < mid <= t[3] <= finish :
            Tmid.append(t)
        elif mid <= t[2] <= t[3] <= finish :
            Taft.append(t)

    # print('----------------')
    # print(f'({invocid})[TPART-B]{Tbef}')
    # print(f'({invocid})[TPART-M]{Tmid}')
    # print(f'({invocid})[TPART-E]{Taft}')
    # print('----------------')

    if Tbef :
        TbefSched = DnCDAGSchedule([t[0] for t in Tbef],M,atg)
        bef1,end1 = getStartFinish(TbefSched)
        Sbef      = DnCRecursive(TbefSched,IS,M,D/3,atg,0,end1)
    else :
        Sbef      = []

    if Tmid :
        Smid      = ScheduleInd2(Tmid,M,D/3,atg)
    else :
        Smid      = []

    if Taft :
        TaftSched = DnCDAGSchedule([t[0] for t in Taft],M,atg)
        bef2,end2 = getStartFinish(TaftSched)
        Saft      = DnCRecursive(TaftSched,IS,M,D/3,atg,0,end2)
    else :
        Saft = []

    # Compute the time for schedule
    _,before_finish = getStartFinish(Sbef)
    _,mid_finish    = getStartFinish(Smid)

    # print('----------------')
    # print(f'({invocid})before_finish:{before_finish},mid_finish:{mid_finish}')
    Smid2 = offsetTask(Smid,before_finish)
    Saft2 = offsetTask(Saft,before_finish+mid_finish)
    # print(f'({invocid}){Sbef}')
    # print(f'({invocid}){Smid2}')
    # print(f'({invocid}){Saft2}')
    # print('----------------')
    
    return (Sbef+Smid2+Saft2)

def DnC(atg,M,D):
    IS       = DnCDAGSchedule(atg.getAllNodes(),M,atg)
    _,finish = getStartFinish(IS)
    S        = DnCRecursive(IS,IS,M,D,atg,0,finish)
    return S

def DnCLike(fl2,D):
    atg = ut.ATG(fl2)
    s1  = time.time()
    S   = DnC(atg,ut.MAXCPU,D)
    s2  = time.time()
    for i,s in enumerate(S) :
        u,m,start,finish = s
        atg.setParamVal(u,'rank',i)
        atg.setParamVal(u,'alloc',m)
        atg.setParamVal(u,'start',start)
        atg.setParamVal(u,'finish',finish)
    
    verpkp,verfinish,maxM,energy = atg.getTotalEtPower()
    return (verpkp,verfinish,maxM,energy,(s2-s1))

def main():
    N   = 100
    id  = 2
    D   = 17*N
    fl2 = f'benchmarks/dag4/random{N}_{id}.dot'
    atg = ut.ATG(fl2)
    S = DnC(atg,ut.MAXCPU,D)
    pprint.pprint(S)
    f   = np.max([u[3] for u in S])
    p,m = ut.computeMaxPkp(S,atg)
    print(f'DnC like with {N} tasks and max {m} cores : {p}, Slack:{D-f}')
    # atg.dumpDot()

if __name__=="__main__":
    main()