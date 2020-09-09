def DnCDAGSchedule(T,M,atg):
    """
        Graham's list scheduling
        Format of IS := <node-id,alloc,start,finish>
        T is just a list of nodes
        T must have the same format as IS
    """
    assert atg.valid, f'DAG not initialized'
    # pprint.pprint(readyT)
    readyM = M      # Free cores
    IS     = []                       # Data structure to hold schedule
    exT    = []                       # Priority queue of executing nodes
    U      = set()                    # Set of already finished nodes
    readyT = atg.getReadyNodes2(T,U)   # Set of enabled nodes
    time   = 0
    i = 0
    while len(T) > 0:
        if len(readyT) > 0 and (readyM > 0):
            # Assign m cores to task-t
            # readyTO = copy.deepcopy(readyT)
            t      = readyT.pop()
            # print(f'Graham({time})|Started:{t}-from-readyT:{readyTO}')
            m      = atg.getPace(t,M)  # WARNING : Are you sure you want to run at pace configuration
            start  = time
            finish = start+atg.getExecutionTime(t,m)
            heapq.heappush(exT,(finish,t,m))
            IS.append((t,m,start,finish))
            readyM -= m
        else :
            if (len(exT) > 0) :
                # exTO = copy.deepcopy(exT)
                finish,t,m = heapq.heappop(exT)
                U.add(t)
                executingNodes = set([u[1] for u in exT])
                readyT = atg.getReadyNodes2(T,U) - executingNodes
                # print(f'Graham({time})|Finished:{t}-from-exT:{exTO},executingNodes:{executingNodes},RnT:{atg.getReadyNodes(T,U)},T:{T},U:{U}')
                # print(f'Graham({time})|UpdateReadyNode:{readyT}')
                # print(f'Graham({time})|removing:{t}')
                T.remove(t)
                readyM += m
                time = finish
            else :
                print(f'Graham({time})|readyT:{readyT},readyM:{readyM}')
                raise ValueError(f'Nobody is executing')
            # print(f'DnCDAGScheduleTTT({time})|readyT:{readyT},T:{T},RnT:{atg.getReadyNodes(T,U)},exT:{exT}')
        # print(f'iter:{i},Total Tasks:{len(T)}')
        i += 1
    return IS



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
