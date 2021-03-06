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
        if not (0 <= readyM <= M) :
            print(f'{readyM} out of bounds at {time}')
    return IS

def ScheduleInd2(T,M,D,atg,id=('mid',0)):
    oldT = NDFH(T,M,D,atg,id)
    uBeg,uFinish = getStartFinish(oldT)
    pkp,maxM,_,_ = ut.computeMaxPkp(oldT,atg)
    # print(f'Id@{id}|MidIndTasks,Size:{len(T)},Start:{uBeg},Finish:{uFinish},maxM:{maxM},pkp:{pkp},deadline:{D}')
    return oldT

def NDFH(T,M,D,atg,id):
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

def DnCRecursive(Tp,IS,M,D,atg,start,finish,depth=0,recurDir='mid'):
    """
        Recursive implementation.
        Format of Tp is same as IS
    """
    if len(Tp) == 0:
        return []
    
    """
        Partition the task-set into three sets.
        The tasks in T_mid do not have any dependences.
    """
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

    # print(f'Id@({recurDir},{depth})|SizeTp:{len(Tp)},SizeChild:<{len(Tbef)},{len(Tmid)},{len(Taft)}>,SizeIS:{len(IS)},start:{start},finish:{finish}')
    # print('----------------')
    # print(f'({invocid})[TPART-B]{Tbef}')
    # print(f'({invocid})[TPART-M]{Tmid}')
    # print(f'({invocid})[TPART-E]{Taft}')
    # print('----------------')

    if Tbef :
        # TbefSched = DnCDAGSchedule([t[0] for t in Tbef],M,atg)
        # bef1,end1 = getStartFinish(TbefSched)
        Sbef      = DnCRecursive(Tbef,IS,M,D/3,atg,start,mid,depth+1,'beg')
    else :
        Sbef      = []

    if Tmid :
        Smid      = ScheduleInd2(Tmid,M,D/3,atg,(recurDir,depth))
    else :
        Smid      = []

    if Taft :
        # TaftSched = DnCDAGSchedule([t[0] for t in Taft],M,atg)
        # bef2,end2 = getStartFinish(TaftSched)
        Saft      = DnCRecursive(Taft,IS,M,D/3,atg,mid,finish,depth+1,'aft')
    else :
        Saft = []

    # Compute the time for schedule
    _,before_finish = getStartFinish(Sbef)
    _,mid_finish    = getStartFinish(Smid)

    # print('----------------')
    # print(f'({invocid})before_finish:{before_finish},mid_finish:{mid_finish}')
    Smid2 = offsetTask(Smid,before_finish)
    Saft2 = offsetTask(Saft,before_finish+mid_finish)
    SU = Sbef+Smid2+Saft2
    uBeg,uFinish = getStartFinish(SU)
    # print(f'({invocid}){Sbef}')
    # print(f'({invocid}){Smid2}')
    # print(f'({invocid}){Saft2}')
    # print('----------------')
    # print(f'Id@({recurDir},{depth})|Exited,Start:{uBeg},Finish:{uFinish}')
    return SU

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
    atg.setScheduled(True)
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