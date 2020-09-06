import math
import time
import copy
import random
import pprint
import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx
import networkx.drawing.nx_agraph as nxdr
import networkx.algorithms as nalgs
import networkx.algorithms.dag as nxdag
import matplotlib.pyplot as plt
import testcvxopt as tcvx
import numpy as np
import itertools as it
import functools as ft
import heapq

BENCH         = ['dfs','cilksort','fib','pi','queens']
L             = 1/0.52
MAXCPU        = 16

def computeMaxPkp(IS,atg):
    """
        Compute the following 
        from the taskset in IS
        1. Finish Time (Start is assumed to be 0.0)
        2. Peak power
        3. Maximum processors used
        4. Total energy consumed
    """
    time   = 0.0
    power  = 0.0
    totalM = 0
    IS.sort(key=lambda u : u[2])
    finishevtQ = []
    allpower   = []
    allM       = []
    allEt      = []
    allEnergy  = []
    for sched in IS:
        # Add up power values
        u,m,start,finish = sched    
        power += atg.getPower(u,m)
        time  += start
        totalM += m
        heapq.heappush(finishevtQ,(finish,atg.getPower(u,m),m))
        allEnergy.append((finish-start)*atg.getPower(u,m))

        # Deduct power values
        if len(finishevtQ) > 0:
            f2,p2,m2 = finishevtQ[0]
            if time >= f2 :
                power -= p2
                totalM -= m2
                heapq.heappop(finishevtQ)
        allpower.append(power)
        allM.append(totalM)
        allEt.append(finish)
        # print(f'iter({i}):{power}')
    if np.max(allM) > MAXCPU :
        # for u in self.G.nodes(data=True) :
        idx = np.argmax(allM)
        print(f'idx:{idx}')
        for i in range(idx-2,idx+2):
            if 0 <= i < len(IS) :
                u,_,_,_ = IS[i]
                pprint.pprint(atg.G.nodes[u])
        


    return np.max(allpower),np.max(allM),np.max(allEt),np.sum(allEnergy)

def combineBench(Nr):
    """
        Create a Cartesian Product of
        benchmarks
    """
    A = []
    for n in range(1,Nr+1):
        B = it.product(BENCH,repeat=n)
        A += [ft.reduce(lambda x,y:x+','+y,bl) for bl in B]
    return A

def generateCombinedBench(Nr):
    A   = combineBench(Nr) 
    etL = lambda x,aet,bet : 1/(aet*x+bet)
    pwL = lambda x,ap,bp : ap*x + bp
    R   = []
    for a in A:
        aL = a.split(',')
        st = len(aL)
        if st > 1:
            etXL = lambda X : np.max([etL(X/st,AETDICT[b],BETDICT[b]) for b in aL])
            pwXL = lambda X : ft.reduce(lambda p1,p2 : p1+p2,[pwL(X/st,APDICT[b],BPDICT[b]) for b in aL])            
            X = np.arange(1,MAXCPU,0.1)
            Y = [1/etXL(x) for x in X]
            Z = [pwXL(x) for x in X]
            aet,bet,r,_,_ = stats.linregress(X,Y)
            ap,bp,r2,_,_ = stats.linregress(X,Z)
            AETDICT[a] = aet
            BETDICT[a] = bet
            APDICT[a] = ap
            BPDICT[a] = bp
            R += [r,r2] 

def equivExecTimeCharacteristics(a1,a2,b1,b2,W1,W2):
    """
        Compute the equivalent execution
        time characteristics of two
        phases with stack sizes
        W1 and W2.
    """
    mu1 = (W1+W2)/W1
    mu2 = (W1+W2)/W2
    a   = L*(a1*a2/(a1*mu2 + a2*mu1))
    b   = L*((a1*b2/mu1) + (a2*b1/mu2))
    return (a,b)

def equivPowerCharacteristics(a1,a2,b1,b2,W1,W2):
    """
        Compute the equivalent power
        characteristics of two
        phases with stack sizes
        W1 and W2.
    """
    mu1 = (W1+W2)/W1
    mu2 = (W1+W2)/W2
    a   = ((a1/mu1)+(a2/mu2))
    b   = b1+b2
    return (a,b)

def extractCharacteristics(bench) :
    """
        Extract execution time
        and power characteristics
    """
    csvPath = f'profile-lace-003/lace-characteristics-{bench}-003.csv'
    df      = pd.read_csv(csvPath)
    x       = df['alloc'].values
    inv_et  = [1/u for u in df['latency(M)'].values]
    p       = df['PKG'].values

    aet,bet,r,_,_ = stats.linregress(x,inv_et)
    tau = lambda m : 1/(aet*m+bet)
    
    ap,bp,r2,_,_ = stats.linregress(x,p)
    rho = lambda m : (ap*m+bp)
    # print(f'B:{bench},r:{r2}')
    return (tau,rho,aet,bet,ap,bp)

AETDICT = dict(zip(BENCH,[extractCharacteristics(b)[2] for b in BENCH]))
BETDICT = dict(zip(BENCH,[extractCharacteristics(b)[3] for b in BENCH]))
APDICT = dict(zip(BENCH,[extractCharacteristics(b)[4] for b in BENCH]))
BPDICT = dict(zip(BENCH,[extractCharacteristics(b)[5] for b in BENCH]))
generateCombinedBench(4)

def generateAllocForChildren(M,N) :
    """
        Divide M cores into N
        components equally.
    """
    if M < N :
        raise ValueError(f'Cannot allocate {M} cores to {N} players')
    alloc = [int(M/N) for _ in range(N)]
    for u in range(M%N) :
        alloc[u] += 1
    return alloc

class ATG(object):
    """
        application task graph.
        Nodes are identified by
        integers
    """
    def __init__(self,dotFile):
        self.G  = nxdr.read_dot(dotFile) # Read a dot file, it is assumed nodes are numbered as str(integers)
        if not nx.is_directed_acyclic_graph(self.G):
            raise IOError(f'The graph is not acyclic')

        N = self.G.nodes(data=True)
        for n in N:
            n[1].update(children=[])
            n[1].update(rank=-1)
            n[1].update(start=-1)
            n[1].update(finish=-1)
            n[1].update(stack=1)
            n[1].update(alloc=-1)
        self.valid = True
        self.isScheduled = False
    
    def setScheduled(self,tv):
        self.isScheduled = True
    
    def __str__(self):
        U = list(self.G.nodes(data=True))
        return pprint.pformat(U)

    def altPrint(self):
        U = dict(list(self.G.nodes(data=True)))
        return ft.reduce(lambda x,y : x+y,[str((k,U[k]['alloc'])) for k in U])

    def dumpDot(self):
        nxdr.write_dot(self.G,f'dump.dot')

    def getAllNodes(self):
        """
            Return the set
            of all nodes 
            and discard their
            property. Used by
            the DnC algorithm
        """
        return set([str(u) for u in self.G.nodes])

    def updateAllocParams(self,u,m,r,start,finish):
        """
            Update the allocation
            of node u
        """
        u = str(u)
        self.G.nodes[u]['alloc'] = m
        self.G.nodes[u]['rank']  = r

    def getExecutionTime(self,u,m):
        """
            Obtain the execution time
            for node u. The cores
            are assumed to divided equally amongst
            children.

            If you have children stacked on
            top of you compute the max execution
            time all of them
        """
        u   = str(u)
        stk = int(self.G.nodes[u]['stack'])
        allocAll = generateAllocForChildren(m,stk)
        if stk == 1 :
            aet = float(self.G.nodes[u]['aet'])
            bet = float(self.G.nodes[u]['bet'])
            m2 = allocAll[0]
            return 1/(aet*m2+bet)
        else :
            et = []
            allChildren = self.G.nodes[u]['children']
            for i,child in enumerate(allChildren):
                aet = float(child[1]['aet'])
                bet = float(child[1]['bet'])
                m2  = allocAll[i]
                et.append(1/(aet*m2+bet))
            return np.max(et)

    def getPower(self,u,m):
        """
            Compute the peak 
            power a node u
            with allocation-m
        """
        u   = str(u)
        stk = int(self.G.nodes[u]['stack'])
        allocAll = generateAllocForChildren(m,stk)
        if stk == 1 :
            ap = float(self.G.nodes[str(u)]['ap'])
            bp = float(self.G.nodes[str(u)]['bp'])
            m2 = allocAll[0]
            return ap*m2 + bp
        else :
            pkp = []
            allChildren = self.G.nodes[u]['children']
            for i,child in enumerate(allChildren):
                ap = float(child[1]['ap'])
                bp = float(child[1]['bp'])
                m2 = allocAll[i]
                pkp.append(ap*m2 + bp)
            return np.sum(pkp)

    def debugPrint(self,prefixStr,u):
        """
            Display the state of node u
        """
        print(prefixStr,end='\t')
        pprint.pprint((u,self.G.nodes[str(u)]))
        print(f'\n')
    
    def mergeNodes(self,v1,v2):
        """
            Merge two nodes
            v1 and v2. Put v2 onto v1
            1. Remove edges of (u,v2)/(v2,u) whenever
            (u,v1)/(v1,u) belongs to the edge set.
            2. updates aet,bet,ap,bp for v1
            3. Removes v2
            4. Update stack parameters
            
            Ignore dynamic parameters like 
            alloc,start,finish and power
        """
        v1 = str(v1)
        v2 = str(v2)
        allE = list(self.G.edges)
        iEv1 = list(self.G.in_edges(v1))
        oEv1 = list(self.G.edges(v1)) # OutEdges incident on v1
        iEv2 = list(self.G.in_edges(v2))
        oEv2 = list(self.G.edges(v2)) # OutEdges incident on v2
        
        # Remove v2 and edges incident on v2
        for e in iEv2 :
            u,_ = e
            allE.remove(e)
            if (not (u,v1) in iEv1) and u != v1:
                allE.append((u,v1))
        for e in oEv2 :
            _,u = e
            allE.remove(e)
            if (not (v1,u) in oEv1) and v1 != u:
                allE.append((v1,u))
        newV = [x for x in self.G.nodes(data=True) if x[0] != v2]

        # Create the new Graph
        H = nx.DiGraph()
        H.add_nodes_from(newV)
        H.add_edges_from(allE)

        # Equivalent benchmark
        benchName = H.nodes[v1]['bench']+','+self.G.nodes[v2]['bench']
        
        # Obtain the children and stacking for
        stackv1 =  int(self.G.nodes[v1]['stack'])
        stackv2 =  int(self.G.nodes[v2]['stack'])
        chld1  =  self.G.nodes[v1]['children']
        chld2  =  self.G.nodes[v2]['children']

        # Obtain benchmark characteristics (Used only during CVX optimization)
        H.nodes[v1]['bench'] = benchName
        H.nodes[v1]['aet'] = AETDICT[benchName]
        H.nodes[v1]['bet'] = BETDICT[benchName]
        H.nodes[v1]['ap'] = APDICT[benchName]
        H.nodes[v1]['bp'] = BPDICT[benchName]
        
        # Merge the child nodes
        # print(f'MergingX34 {v2} onto {v1}|stackv1:{stackv1},stackv2:{stackv2}')
        if stackv1 == 1 and stackv2 == 1: # No previous stacked children neither in v1 or v2
            H.nodes[v1]['children'] = [(v2,self.G.nodes[v2])] + [(v1,self.G.nodes[v1])]
            H.nodes[v1]['stack'] = 2
        elif stackv2 == 1 : # v1 has non-empty children
            H.nodes[v1]['children'] = chld1 + [(v2,self.G.nodes[v2])]
            H.nodes[v1]['stack'] = 1 + stackv1
        elif stackv1 == 1 : # v2 has non-empty children
            H.nodes[v1]['children'] = chld2 + [(v1,self.G.nodes[v1])]
            H.nodes[v1]['stack'] = 1 + stackv2
        else : # Both have non-empty children
            H.nodes[v1]['children'] = chld1 + chld2
            H.nodes[v1]['stack'] = stackv1 + stackv2
        
        # Reset the graph
        self.G = H
        # if v1 == '16' or v1 == '19' or v1 == '17':
        #     self.debugPrint('mergStep',v1)
        
    def computeDiff(self,u,v,m):
        """
            Estimate the peformance
            difference in execution
            when tasks are 
            (u,v) are executed 
            in serially, whith their default
            allocation vs when they are
            executed concurrently with m cores.
        """
        u  = str(u)
        v  = str(v)
        et = lambda m,a,b : 1/(a*m+b)
        a1 = float(self.G.nodes[u]['aet'])
        b1 = float(self.G.nodes[u]['bet'])
        a2 = float(self.G.nodes[v]['aet'])
        b2 = float(self.G.nodes[v]['bet'])
        m1 = int(self.G.nodes[u]['alloc'])
        m2 = int(self.G.nodes[v]['alloc'])
        W1 = int(self.G.nodes[u]['stack'])
        W2 = int(self.G.nodes[v]['stack'])
        mu1 = (W1+W2)/W1
        mu2 = (W1+W2)/W2
        a,b = equivExecTimeCharacteristics(a1,a2,b1,b2,W1,W2)

        tu  = et(m1,a1,b1)
        tv  = et(m2,a2,b2)
        tuv = et(m,a,b)
        err = np.max([et(m/mu1,a1,b1),et(m/mu2,a2,b2)])-tuv

        # Ideally tuv < (tu+tv)
        return (tuv-(tu+tv),err)

    def computeAC(self):
        """
            compute the transitive closure of
            the DAG
        """
        # Compute some structural properties apriori
        H = list(nxdag.antichains(self.G))
        # nxdr.write_dot(self.G,f'demo.dot')
        A = [tuple(h) for h in H if (len(h) == 2) and (int(self.G.nodes[h[0]]['stack']) <= 4) and (int(self.G.nodes[h[1]]['stack']) <= 4)]
        return A

    def getWidth(self):
        # print(f'ATG width:{self.W}')
        H = list(nxdag.antichains(self.G))
        W = np.max([len(h) for h in H])
        return W

    def computeBestPair(self,m):
        """
            Obtain a pair of AC tasks
            which when stacked yields the
            greatest reduction in execution 
            time. 
            
            Must stop when the stacking factor
            is already 4.
        """
        A   = self.computeAC()
        if len(A) == 0:
            return None
        else:
            B   = [-(self.computeDiff(u,v,m)[0]) for (u,v) in A]
            idx = np.argmax(B)
            return A[idx]

    def topoSort(self):
        T = [n for n in nxdag.topological_sort(self.G)]
        # Update the order
        for i,t in enumerate(T):
            self.G.nodes[t]['rank'] = i
        return T
    
    def getNodeParams(self,u):
        u   = str(u) 
        aet = float(self.G.nodes[u]['aet'])
        bet = float(self.G.nodes[u]['bet'])
        ap  = float(self.G.nodes[u]['ap'])
        bp  = float(self.G.nodes[u]['bp'])
        llim = float(self.G.nodes[u]['stack'])
        return (aet,bet,ap,bp,llim)

    def getParamVal(self,u,param):
        u     = str(u)
        param = str(param)
        return self.G.nodes[u][param]
    
    def setParamVal(self,u,param,val):
        u     = str(u)
        param = str(param)
        self.G.nodes[u][param] = val

        # Some parameter changes require update to children as well
        if param == 'rank':
            rank = val
            # print(f'setting rank for {u} = {rank}')
            for child in self.G.nodes[u]['children'] :
                child[1][param] = rank
        if param == 'alloc':
            allocTotal = val
            stk = int(self.G.nodes[u]['stack'])
            allocAll = generateAllocForChildren(allocTotal,stk)
            # if (u == '16') :
            #     print(f'u:{u},AllocAll:{allocAll},Stack:{stk}')
            if stk == 1 :
                self.G.nodes[u][param] = allocAll[0]
            else :
                self.G.nodes[u][param] = allocTotal # np.sum(allocAll)
                allChildren = self.G.nodes[u]['children']
                for i,child in enumerate(allChildren) :
                    child[1][param] = allocAll[i]
        if param == 'start':
            start = val
            self.G.nodes[u][param] = start
            for child in self.G.nodes[u]['children'] :
                child[1][param] = start
        if param == 'finish':
            finish = val  # This is value is useless
            stk = int(self.G.nodes[u]['stack'])
            finishALL = []
            # Different member of the stack will have different finish times
            for child in self.G.nodes[u]['children'] :
                alloc  = int(child[1]['alloc'])
                start  = int(child[1]['start'])
                aet    = float(child[1]['aet'])
                bet    = float(child[1]['bet'])
                finish2 = start + 1/(aet*alloc+bet)
                child[1]['finish'] = finish2
                finishALL.append(finish2)
            if stk > 1 :
                self.G.nodes[u][param] = np.max(finishALL)
            else :
                self.G.nodes[u][param] = finish
                
    def getPace(self,u,M):
        """
            Get the minimum energy
            allocation for node-u.
            Used by the DnC algorithm
        """
        idx = np.argmin([self.getPower(u,m)*self.getExecutionTime(u,m) for m in range(1,M+1)])
        return idx+1

    def getReadyNodes(self,T,U) :
        """
            Find all such nodes in 
            T, all of whose predecessors
            lie in U
        """
        return self.getReadyNodes2(T,U)

    def getReadyNodes2(self,T,U) :
        """
            Find the subset of T, whose
            predecessors are in U and which
            is not already in U
        """ 
        readyNodes = set()
        for t in T :
            pred = set(self.G.predecessors(t)) & set(T)
            if (pred <= U) and (t not in U):
                readyNodes.add(t)
        return readyNodes

    def getMinAlloc(self,u,D,M):
        """
            Get the minimum allocation of
            a phase that meets the deadline D
        """
        u = str(u)
        i = 1
        while True:
            if self.getExecutionTime(u,i) <= D :
                break
            else :
                i = i+1
        if i >= M :
            return M
        else :
            return i
    
    def getTotalEtPower(self):
        """
            Compute the total
            execution and
            peak power for a schedule
        """
        IS2 = self.checkAllocationCorrectness()
        IS = [(k,u['alloc'],u['start'],u['finish']) for k,u in IS2.items()]
        maxpkp,maxM,et,energy=computeMaxPkp(IS,self)
        return (maxpkp,et,maxM,energy)

    def checkAllocationCorrectness(self):
        """
            Verify if the precedence and allocation
            constraints are satisfied. Does not
            check for the violation of resource 
            constraints.
        """
        IS = dict()
        for u in self.G.nodes(data=True) :
            if self.setScheduled :
                """
                    Ensure that a scheduled
                    ATG has all its parameters
                    set correctly
                """
                allParamSet = (u[1]['rank'] >= 0) and \
                              (u[1]['stack'] > 0) and \
                              (u[1]['alloc'] > 0) and \
                              (u[1]['start'] >= 0) and \
                              (u[1]['finish'] >= u[1]['start'])
                if not allParamSet :
                    raise ValueError(f'Parameters for {pprint.pformat(u)} not correctly set')
            if int(u[1]['stack']) == 1 :
                IS[u[0]] = {
                    'rank' : u[1]['rank'],
                    'alloc' : u[1]['alloc'],
                    'start' : u[1]['start'],
                    'finish' : u[1]['finish']
                }
            else :
                for v in u[1]['children'] :
                    IS[v[0]] = {
                        'rank' : v[1]['rank'],
                        'alloc' : v[1]['alloc'],
                        'start' : v[1]['start'],
                        'finish' : v[1]['finish']
                    }
        # Verify precedence constraints
        for (u,v) in self.G.edges() :
            if IS[u]['finish'] > IS[v]['start'] :
                print(self)
                raise ValueError(f'Dep not satisfied for {(u,v)}')
        return IS
    
    def initalloc(self,D):
        """
            Start with an initial allocation
            All nodes serialized and allocated 
            the maximum cores
        """
        T     = self.topoSort()
        start  = 0.0
        maxpkp = 0.0
        finish = 0.0
        for r,t in enumerate(T):
            alloc = MAXCPU
            self.setParamVal(t,'rank',r)
            self.setParamVal(t,'alloc',alloc)
            finish = start + self.getExecutionTime(t,alloc)
            self.setParamVal(t,'start',start)
            self.setParamVal(t,'finish',finish)
            maxpkp = np.max([maxpkp,self.getPower(t,alloc)])
            start = finish
        return (maxpkp,finish,-1)
    
    def cvxalloc(self,D):
        """
            Compute the allocations of
            nodes of atg, which is (topologically)
            sorted as in T.
        """
        T   = self.topoSort()
        aet = []
        bet = []
        ap  = []
        bp  = []
        llim = []
        N   = len(T)         # Number of phases, also must match the length of the arrays declared before
        for r,t in enumerate(T):
            a1,b1,a2,b2,llim1 = self.getNodeParams(t)
            aet.append(a1)
            bet.append(b1)
            ap.append(a2)
            bp.append(b2)
            llim.append(llim1)
            # self.setParamVal(t,'rank',r)
        
        opt = tcvx.CPPCVXOptimizer()
        x   = [1 for _ in range(N)] + [0.0]
        opt.setParams(N,x,aet,bet,ap,bp,llim,MAXCPU)
        opt.optimize(D)
        xopt = opt.getOpt()
    
        # Discretize and allocate
        start  = 0.0
        finish = 0.0
        maxpkp = 0.0
        xopt2  = []
        for r,t in enumerate(T) :
            self.setParamVal(t,'rank',r)
    
            # Allocation for myself. Allocation for my children is done within setParamVal
            allocAllStack = math.ceil(xopt[r])
            self.setParamVal(t,'alloc',allocAllStack)
    
            finish = start + self.getExecutionTime(t,allocAllStack)
            self.setParamVal(t,'start',start)
            self.setParamVal(t,'finish',finish)
            start = finish
            maxpkp = np.max([maxpkp,self.getPower(t,allocAllStack)])
            xopt2.append(allocAllStack)
        
        # self.debugPrint('cvxAlloc','19')
        # self.debugPrint('cvxAlloc','16')
        return (maxpkp,finish,-1)


if __name__=="__main__":
    generateCombinedBench(4)