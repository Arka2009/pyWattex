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






