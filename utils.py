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
import numpy as np
import testcvxopt as tcvx
import itertools as it
import functools as ft

BENCH         = ['dfs','cilksort','fib','pi','queens']
L             = 1/0.52
MAXCPU        = 16

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
        
        self.valid    = True
    
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

    # def getReadyNodes(self,T,U) :
    #     pass

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
            with m cores for node u.

            If you have children stacked on
            top of you compute the execution
            time and max all of them
        """
        et  = []
        u   = str(u)
        for child in self.G.nodes[u]['children']:
            aet = float(child[1]['aet'])
            bet = float(child[1]['bet'])
            m2  = int(child[1]['alloc'])
            et.append(1/(aet*m2+bet))
        aet = float(self.G.nodes[u]['aet'])
        bet = float(self.G.nodes[u]['bet'])
        et.append(1/(aet*m+bet))
        return np.max(et)

    def getPower(self,u,m):
        """
            Compute the peak 
            power a node u
            with allocation-m
        """
        ap = float(self.G.nodes[str(u)]['ap'])
        bp = float(self.G.nodes[str(u)]['bp'])
        p  = ap*m + bp
        return p

    def mergeNodes(self,v1,v2):
        """
            Merge two nodes
            v1 and v2. 
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
        
        # Remove the edges and nodes
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
        H.nodes[v1]['bench'] += ','+self.G.nodes[v2]['bench']
        benchName = H.nodes[v1]['bench']
        
        # Append the benchmark details
        stack1 =  int(self.G.nodes[v1]['stack'])
        stack2 =  int(self.G.nodes[v2]['stack'])
        chld1  =  self.G.nodes[v1]['children']
        chld2  =  self.G.nodes[v2]['children']

        H.nodes[v1]['aet'] = AETDICT[benchName]
        H.nodes[v1]['bet'] = BETDICT[benchName]
        H.nodes[v1]['ap'] = APDICT[benchName]
        H.nodes[v1]['bp'] = BPDICT[benchName]
        
        H.nodes[v1]['stack'] = stack1 + stack2
        H.nodes[v1]['children'] = chld1 + chld2 + [(v2,self.G.nodes[v2])]
        # Reset the graph
        self.G = H
        

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
        A = [tuple(h) for h in H if len(h) == 2]
        return A

    def getWidth(self):
        # print(f'ATG width:{self.W}')
        H = list(nxdag.antichains(self.G))
        W = np.max([len(h) for h in H])
        return W

    def computeBestPair(self,m):
        """
            Obtain a pair of AC tasks
            which when stack yields the
            greatest reduction in execution 
            time
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

        return (aet,bet,ap,bp)

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
            for n in self.G.nodes[u]['children'] :
                n[1][param] = rank
        if param == 'alloc':
            alloc = val
            stack = int(self.G.nodes[u]['stack'])
            self.G.nodes[u][param] = int(alloc)
            # print(f'setting alloc for {u} = {int(alloc)}')
            for n in self.G.nodes[u]['children'] :
                n[1][param] = int(alloc/stack)
        if param == 'start':
            start = val
            for n in self.G.nodes[u]['children'] :
                n[1][param] = start
        if param == 'finish':
            # Different member of the stack will have different finish times
            for n in self.G.nodes[u]['children'] :
                alloc  = int(n[1]['alloc'])
                start  = int(n[1]['start'])
                aet    = float(n[1]['aet'])
                bet    = float(n[1]['bet'])
                finish = start + 1/(aet*alloc+bet)
                n[1]['finish'] = finish
   
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
        # readyNodes = set()
        # T2 = set(T)
        # for t in T2 :
        #     pred = set(self.G.predecessors(t))
        #     if pred <= U : 
        #         if t in (T2-U) :
        #             readyNodes.add(t)
        # return readyNodes
        nbdT = (ft.reduce(lambda u,v : u|v,[set(self.G.predecessors(t)) | set([t]) for t in T]) & set(T)) - U
        return nbdT

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
            peak power
        """
        et  = 0.0
        pkp = 0.0
        startL = []
        finishL = []
        pkpL= []
        for u in self.G.nodes(data=True):
            m = u[1]['alloc']
            startL.append(u[1]['start'])
            finishL.append(u[1]['finish'])
            pkpL.append(self.getPower(u[0],m))

        et = np.max(finishL) - np.min(startL)
        pkp = np.max(pkpL)
        return (pkp,et)

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
    ub  = [MAXCPU for _ in range(N)]+[float('inf')]

    opt = tcvx.CPPCVXOptimizer()
    x   = [1 for _ in range(N)]+[0.0]
    opt.setParams(N,x,a1,b1,a2,b2,1,MAXCPU)
    opt.optimize(D)
    xopt = opt.getOpt()
    print(xopt[:-1])

def cvxalloc(atg,D):
    """
        Compute the allocations of
        nodes of atg, which is (topologically)
        sorted as in T.
    """
    T   = atg.topoSort()
    aet = []
    bet = []
    ap  = []
    bp  = []
    N   = len(T)         # Number of phases, also must match the length of the arrays declared before
    for r,t in enumerate(T):
        a1,b1,a2,b2 = atg.getNodeParams(t)
        aet.append(a1)
        bet.append(b1)
        ap.append(a2)
        bp.append(b2)
        # atg.setParamVal(t,'rank',r)
    
    opt = tcvx.CPPCVXOptimizer()
    x   = [1 for _ in range(N)] + [0.0]
    opt.setParams(N,x,aet,bet,ap,bp,1,MAXCPU)
    opt.optimize(D)
    xopt = opt.getOpt()

    # Discretize and allocate
    start  = 0.0
    finish = 0.0
    maxpkp = 0.0
    xopt2  = []
    for r,t in enumerate(T) :
        atg.setParamVal(t,'rank',r)
        stack = int(atg.getParamVal(t,'stack'))
        alloc = (stack)*(math.ceil(xopt[r]/(stack)))  # Allocation for the entire stack
        # print(f'cvxalloc({t})|stack:{stack},alloc:{alloc},xopt:{xopt[r]}')
        atg.setParamVal(t,'alloc',alloc)
        finish = start + atg.getExecutionTime(t,alloc)
        atg.setParamVal(t,'start',start)
        atg.setParamVal(t,'finish',finish)
        start = finish
        maxpkp = np.max([maxpkp,atg.getPower(t,alloc)])
        xopt2.append(alloc)
    
    return maxpkp,finish

def initalloc(atg,D):
    """
        Start with an initial allocation
        All nodes serialized and allocated 
        the maximum cores
    """
    T     = atg.topoSort()
    start  = 0.0
    maxpkp = 0.0
    finish = 0.0
    for r,t in enumerate(T):
        alloc = MAXCPU
        atg.setParamVal(t,'rank',r)
        atg.setParamVal(t,'alloc',alloc)
        finish = start + atg.getExecutionTime(t,alloc)
        atg.setParamVal(t,'start',start)
        atg.setParamVal(t,'finish',finish)
        maxpkp = np.max([maxpkp,atg.getPower(t,alloc)])
        start = finish
    return maxpkp,finish

def PkMin(fl2,D,debugPrint=False):
    # oldpkp = 0.0
    pkp    = 0.0 
    atg = ATG(fl2)
    tol = 1e-8
    s1  = time.time()
    i = 0
    pkp,finish = initalloc(atg,D)
    completelySerialized = False
    
    while(not completelySerialized):
        if i == 0 :
            oldatg     = copy.deepcopy(atg)
            oldpkp     = pkp
            oldfin     = finish
        else :
            pkp1,finish1 = atg.getTotalEtPower()
            if pkp1 < oldpkp and finish1 <= D: # Save the best encountered allocation
                oldatg     = copy.deepcopy(atg)
                oldpkp     = pkp
                oldfin     = finish
        
        if debugPrint :
            pkp2,finish2 = atg.getTotalEtPower()
            print(f'iter_Beg@{i}|Et:{finish2},Pkp:{pkp2},ATG:{atg}\n')

        # CVX Opt Step
        pkp_cvx,finish_cvx = cvxalloc(atg,D)
        if (pkp_cvx < oldpkp) and (finish_cvx <= D) : # Save the best encountered allocation
            oldatg  = copy.deepcopy(atg)
            oldpkp  = pkp_cvx
            oldfin  = finish_cvx
        
        if debugPrint :
            pkp2,finish2 = atg.getTotalEtPower()
            print(f'iter_AftCVXOpt@{i}|Et:{finish2},Pkp:{pkp2},ATG:{atg}\n')

        # Deadline Missed
        # if finish > D :
        #     print(f'STOPPED : DEADLINE MISS')
        #     break
        # if oldpkp < pkp :
        #     print(f'STOPPED : PREVIOUS CONFIG BETTER')
        #     break

        # DAG merging
        ac = atg.computeBestPair(MAXCPU)
        if ac:
            a,c = ac
            atg.mergeNodes(a,c)
        else :
            completelySerialized = True

        # AFter DAG merging
        # if debugPrint :
        #     pkp2,finish2 = atg.getTotalEtPower()
        #     print(f'iter_AftDAGMerging@{i}|Et:{finish2},Pkp:{pkp2},ATG:{atg}\n')
        
        i += 1

    s2  = time.time()
    # Verfication
    verpkp, verfinish = oldatg.getTotalEtPower()
    print(f'Best Configuration|Et:{verfinish},Pkp:{verpkp}')
    return oldpkp,oldfin,(s2-s1),oldatg

if __name__=="__main__":
    generateCombinedBench(4)

