PythonAny = any
PythonMin, PythonMax = min, max
from numpy import isnan, array, atleast_1d, all, searchsorted, logical_or, any, nan, \
vstack, inf, where, logical_not, min, abs, insert, logical_xor, argsort, zeros_like

# for PyPy
from openopt.kernel.nonOptMisc import isPyPy

try:
    from numpy import append
except ImportError:
    def append(*args, **kw):
        raise ImportError('function append() is absent in PyPy yet')
        
from interalgLLR import *

try:
    from bottleneck import nanmin, nanmax
except ImportError:
    from numpy import nanmin, nanmax

hasPoint = lambda Lx, Ux, point:\
    True if Lx.size != 0 and any([(all(Lx[i]<=point) and all(Ux[i]>=point)) for i in range(Lx.shape[0])]) else False
pointInd = lambda Lx, Ux, point:\
    where([(all(Lx[i]<=point) and all(Ux[i]>=point)) for i in range(Lx.shape[0])])[0].tolist()

a = np.array([-0.39])


def processBoxes(p, nlhc, residual, definiteRange, Lx, Ux, ooVars, fd_obj, C, 
                 bestKnownObjValueWithPoint, cutLevel, nNodes, 
                 bestKnownObjValueWithoutPoint, fTol, Solutions, varTols, inactiveNodes, dataType,
                 maxNodes, semiinvariant, indTC, xRecord, activeCons):
    assert Lx.shape[0] == semiinvariant.size
#    print('!', p.iter, hasPoint(Lx, Ux, a))
    isSNLE = p.probType in ('NLSP', 'SNLE')
#    print('iter:',p.iter,'bestKnownObjValueWithoutPoint:',bestKnownObjValueWithoutPoint)

    #debug!!!!!!!!!!
#    if len(inactiveNodes) == 0 and Lx.size == 0:
#        print('no points left')
#    else:
#        _lx, _ux = vstack([node.Lx for node in inactiveNodes]+[Lx]), vstack([node.Ux for node in inactiveNodes]+[Ux])
#        if p.iter>=197: #not hasPoint(_lx, _ux, a):
#            print('iter>>', p.iter, 'in start of preocessBoxes:', hasPoint(_lx, _ux, a))

    
#    print p.iter, Lx, Ux, hasPoint(Lx, Ux, [10.,  20.,  10.,  50])

    #maxSolutions, solutions, coords = Solutions.maxNum, Solutions.solutions, Solutions.coords
    maxSolutions, solutions = Solutions.maxNum, Solutions.solutions

    threshold_prev = \
    float(0 if isSNLE else PythonMin((bestKnownObjValueWithoutPoint, 
    bestKnownObjValueWithPoint - (fTol if maxSolutions == 1 else 0))))
    threshold_prev  =  PythonMin(1e300, threshold_prev)
    
    if 0 and isSNLE:
        Uf = residual
        Lf = zeros_like(Uf)
        Lf2, Uf2, bestKnownObjValueWithoutPoint = getFdata(Lx, Ux, ooVars, p, fd_obj, dataType, 
        bestKnownObjValueWithoutPoint, nlhc)
#        print p.iter,nanmax(Uf/Uf2) / nanmin(Uf/Uf2)
#        Lf, Uf = Lf2, Uf2
        Uf = Uf2
        Lf = zeros_like(Uf)
    else:
        Lx, Ux, Lf, Uf, bestKnownObjValueWithoutPoint, semiinvariant, indTC = \
        getFdata(Lx, Ux, ooVars, p, fd_obj, dataType, bestKnownObjValueWithoutPoint, nlhc, 
        semiinvariant, indTC)
        #assert semiinvariant.size == Lx.shape[0]
        
#        print(p.iter, Lf, Uf)
    
    assert len(activeCons) == semiinvariant.size == Lx.shape[0]
    
    if Lf is None:
        return inactiveNodes, cutLevel, threshold_prev, semiinvariant, Solutions, xRecord, bestKnownObjValueWithoutPoint, bestKnownObjValueWithPoint
    
    
    # was moved to interalgLLR.formNodes()
#    Lx, Ux, Lf, Uf, semiinvariant, indTC, nlhc, residual, activeCons = \
#    remove_NaN_nodes(Lx, Ux, Lf, Uf, semiinvariant, indTC, nlhc, residual, activeCons)    

    
    
#    if p.iter > 9:
#        print('3>>', p.iter, hasPoint(Lx, Ux, a))

#    #debug
#    if p.iter > 116:
#        N = inactiveNodes
#        print('len(N):', len(N))
#        print('m:', Lx.shape[0])
##        N = AllNodes
#        active_cons = [node.activeCons for node in N] + activeCons
#        cons_counter = p._constraints_counter
#        c2 = dict()
#        for _cons_list in active_cons:
#            for elem in _cons_list:
#                tmp = c2.get(elem, None)
#                if tmp is None:
#                    c2[elem] = 1
#                else:
##                    tmp += 1
#                    c2[elem] += 1
#        print(sum(c2.values()), sum(cons_counter.values()))
#        print('-----------')

    
    nodes, Lx, Ux, Lf, Uf, semiinvariant, indT, nlhc, residual, activeCons = formNodes(Lx, Ux, nlhc, indTC, residual, Lf, Uf, semiinvariant, p, activeCons)
    assert len(activeCons) == semiinvariant.size == Lx.shape[0]
    #nodes, cutLevel = removeSomeNodes(nodes, threshold_prev, cutLevel, p)
    #Lx, Ux = TruncateSomeBoxes(Lx, Ux, Lf, Uf, threshold)
    if len(nodes) == 0:# can be after removing nodes with nan
        return inactiveNodes, cutLevel, threshold_prev, semiinvariant, Solutions, xRecord, bestKnownObjValueWithoutPoint, bestKnownObjValueWithPoint
    
#    print('asdf')
    if p.solver.dataHandling == 'raw':
        if not isSNLE:
            if 0:
                tmp = Lf.copy()
                tmp[tmp > threshold_prev] = -inf
                M = atleast_1d(nanmax(tmp, 1))
                assert len(nodes) == M.size
                for i, node in enumerate(nodes):
                    node.th_key = M[i]
                    node.threshold = threshold_prev       
            else:
#                print('asdf')
                tmp = Lf.copy()
                
                # TODO: check is it ever trigger in current interalg implementation
                tmp[tmp > threshold_prev] = 1e300#-inf
                
                # TODO: mb change Uf & Lf for this ind to 1e300 as well?
                half_width = tmp.shape[1]//2
                Lf_l,  Lf_u = tmp[:, :half_width], Lf[:, half_width:]
                tmp = where((Lf_l < Lf_u) | isnan(Lf_u), Lf_l, Lf_u)
                
                M = atleast_1d(nanmax(tmp, 1))
                assert len(nodes) == M.size
#                ind_exclude = []
                for i, node in enumerate(nodes):
#                    if M[i] == 1e300:
#                        ind_exclude.append(i)
#                        continue
                    node.th_key = M[i]
                    node.threshold = threshold_prev
#                    print(p.iter,'th_key:', M[i], 'threshold_prev:', threshold_prev)

#                for i in ind_exclude[::-1]: 
#                    nodes.pop(i)


        # TODO: MB HANDLE IT INSIDE formNodes() FUNC
        if nlhc is None:#TODO: mb rework/improvement for SNLE 
            for node in nodes: 
                node.tnlh_fixed = node.nlh_obj_fixed 
        else:
            for node in nodes: 
#                print('!', p.iter, node.nlh_obj_fixed ,  node.nlhc )
                node.tnlh_fixed = node.nlh_obj_fixed + node.nlhc 
                
            
        AllNodes = nodes + inactiveNodes
        
        
        #tnlh_fixed = vstack([node.tnlh_fixed for node in AllNodes])
        tnlh_fixed = vstack([node.tnlh_fixed for node in nodes])
        
        #tnlh_fixed_local = vstack([node.nlh_obj_fixed for node in nodes])#tnlh_fixed[:len(nodes)]
#        if isSNLE:
#            tnlh_curr = tnlh_fixed_local#vstack([node.nlhc for node in nodes])
        if 1:
            tmp_u = where(Uf>threshold_prev, threshold_prev, Uf)#Uf.copy()
            tmp_l = where(Lf==-inf, -1e300, Lf)
#            tmp[tmp>threshold_prev] = threshold_prev
            tmp2 = tmp_u - tmp_l
#            ind_inf = tmp2==inf

            tmp2[tmp2<1e-300] = 1e-300
#            tmp2[tmp2<p.fTol] = 1e-300
            
            Lf_exclude_ind = Lf > threshold_prev
            if Lf_exclude_ind.any():
                tmp2[Lf_exclude_ind] = nan
                if not isSNLE:
                    cutLevel = PythonMin(cutLevel, min(Lf[Lf_exclude_ind]))
            #tnlh_curr = tnlh_fixed_local - log2(tmp2)
            tnlh_curr = tnlh_fixed - log2(tmp2)
            
#            tnlh_curr[ind_inf] = 1e300
        else:
            if isSNLE:
                tnlh_curr = tnlh_fixed_local
            else:
                tmp = Uf.copy()
                tmp[tmp>threshold_prev] = threshold_prev
                tmp2 = tmp - Lf
                tmp2[tmp2<1e-300] = 1e-300
                tmp2[Lf > threshold_prev] = nan
                tnlh_curr = tnlh_fixed_local - log2(tmp2)
                
        tnlh_curr_best = nanmin(tnlh_curr, 1)
        
        for i, node in enumerate(nodes):
            node.tnlh_curr = tnlh_curr[i]
            node.tnlh_curr_best = tnlh_curr_best[i]
        
        # TODO: use it instead of code above
        #tnlh_curr = tnlh_fixed_local - log2(where() - Lf)
    else:
        tnlh_curr = None
    
    
    
    # TODO: don't calculate PointVals for zero-p regions
    PointVals, PointCoords = getCentersValues(ooVars, Lx, Ux, activeCons, tnlh_curr, fd_obj, C, p.contol, dataType, p) 
    if PointVals.size != 0:
#        print('case1')
        xk, Min = getBestCenterAndObjective(PointVals, PointCoords, dataType)
    else: # all points have been removed by remove_NaN_nodes
#        print('case2')
        xk = p.xk
        Min = nan
        
    if bestKnownObjValueWithPoint > Min:
        bestKnownObjValueWithPoint = Min
        xRecord = xk.copy()# TODO: is copy required?
#    print('Min:',Min)
    bestKnownObjValueWithoutPoint = nanmin((Min, bestKnownObjValueWithoutPoint))
    
    threshold = \
    float(0 if isSNLE else PythonMin(bestKnownObjValueWithoutPoint, bestKnownObjValueWithPoint - (fTol if maxSolutions == 1 else 0)))

    tmp = array([node.key for node in (nodes + inactiveNodes)])
    f_bound_estimation = tmp.min()
    
    if isfinite(bestKnownObjValueWithPoint):
        p.f_bound_distance = bestKnownObjValueWithPoint - f_bound_estimation
    if p.goal in ('max', 'maximum'):
        f_bound_estimation = -f_bound_estimation
    p.f_bound_estimation = f_bound_estimation
    
#    print '!', p.iter, Lx, Ux, hasPoint(Lx, Ux, [10.,  20.,  10.,  50])

    assert len(activeCons) == semiinvariant.size == Lx.shape[0]

    if p.solver.dataHandling == 'raw':
        
        if threshold != threshold_prev and not isSNLE:
            thresholds = array([node.threshold for node in AllNodes])
            
            #prev
            #ind_update = where(thresholds > threshold + 0.01* fTol)[0]
            
            #new
            th_keys = array([node.th_key for node in AllNodes])
            delta_thresholds = thresholds - threshold
            ind_update = where(10 * delta_thresholds > thresholds - th_keys)[0]
            
            update_nlh = ind_update.size != 0 
#                  print 'Lf MB:', float(Lf_tmp.nbytes) / 1e6
#                  print 'percent:', 100*float(ind_update.size) / len(AllNodes) 

# !!!!!!!!!!!!!!!!!!!!
# TODO: check and enable updateNodes()
# !!!!!!!!!!!!!!!!!!!!
            #temporary disabled
            if 1 and update_nlh:
                #nodesToUpdate = AllNodes[ind_update]
                nodesToUpdate = [AllNodes[i] for i in ind_update]
#                    from time import time
#                    tt = time()
                updateNodes(nodesToUpdate, threshold, p.fTol)
#                    if not hasattr(p, 'Time'):
#                        p.Time = time() - tt
#                    else:
#                        p.Time += time() - tt
            
            tmp = array([node.key for node in AllNodes])
            #print('iter:', p.iter, 'tmp:', tmp)
            cond = tmp > threshold
            IND = where(cond)[0]
            cutLevel = PythonMin([AllNodes[i].key for i in IND]+[cutLevel])
            ind_remain = where(logical_not(cond))[0]
            if p.debug and ind_remain.size != len(AllNodes):
                p.debugmsg('num excluded by fTol: %d from %d' % (len(AllNodes)-ind_remain.size, len(AllNodes)))
            AllNodes = [AllNodes[i] for i in ind_remain]

        NN = atleast_1d([node.tnlh_curr_best for node in AllNodes])
        IND = logical_or(isnan(NN), NN == inf)

        if any(IND):
            ind = where(logical_not(IND))[0]
            AllNodes = [AllNodes[i] for i in ind]#AllNodes[ind]
            #tnlh = take(tnlh, ind, axis=0, out=tnlh[:ind.size])
            #NN = take(NN, ind, axis=0, out=NN[:ind.size])
            NN = NN[ind]

        #
        if 1 or not isSNLE or p.maxSolutions == 1:
            astnlh = argsort(NN)
            AllNodes = [AllNodes[i] for i in astnlh]#AllNodes[astnlh]
        

        
        
        
#            if isPyPy:
#                AllNodes = [AllNodes[i] for i in astnlh]#AllNodes[astnlh]
#            else:
#                # TODO: 
#                AllNodes = AllNodes[astnlh]
            
#        print(AllNodes[0].nlhc, AllNodes[0].tnlh_curr_best)
        # Changes
#        if NN.size != 0:
#            ind = searchsorted(NN, AllNodes[0].tnlh_curr_best+1)
#            tmp1, tmp2 = AllNodes[:ind], AllNodes[ind:]
#            arr = [node.key for node in tmp1]
#            Ind = argsort(arr)
#            AllNodes = hstack((tmp1[Ind], tmp2))
        #print [node.tnlh_curr_best for node in AllNodes[:10]]
    
    else: #if p.solver.dataHandling == 'sorted':
        if isSNLE and p.maxSolutions != 1: 
            AllNodes = nodes + inactiveNodes#hstack((nodes, inactiveNodes))
        elif isPyPy:
            AllNodes = nodes + inactiveNodes
            AllNodes.sort(key = lambda obj: obj.key)
        else:
            nodes.sort(key = lambda obj: obj.key)

            if len(inactiveNodes) == 0:
                AllNodes = nodes
            else:
                arr1 = [node.key for node in inactiveNodes]
                arr2 = [node.key for node in nodes]
                IND = searchsorted(arr1, arr2)
                AllNodes = insert(inactiveNodes, IND, nodes).tolist()
#                if p.debug:
#                    arr = array([node.key for node in AllNodes])
#                    #print arr[0]
#                    assert all(arr[1:]>= arr[:-1])
    
    
    
    if maxSolutions != 1:
        Solutions = processCandidates(PointCoords, PointVals, fTol, varTols, Solutions)
        
        p._nObtainedSolutions = len(solutions)
        if p._nObtainedSolutions > maxSolutions:
            solutions = solutions[:maxSolutions]
            p.istop = 0
            p.msg = 'user-defined maximal number of solutions (p.maxSolutions = %d) has been exeeded' % p.maxSolutions
            return AllNodes, cutLevel, threshold, None, Solutions, xRecord, bestKnownObjValueWithoutPoint, bestKnownObjValueWithPoint
    
#    print('bestKnownObjValueWithPoint:',bestKnownObjValueWithPoint)
    p.iterfcn(xRecord, bestKnownObjValueWithPoint)

    # rTol processing
    if 1 and not isSNLE and isfinite(bestKnownObjValueWithoutPoint) and len(AllNodes):
        nodes_infinum = array([node.key for node in AllNodes])
        abs_nodes_infinum = abs(nodes_infinum)
        abs_bestKnownObjValueWithoutPoint = abs(bestKnownObjValueWithoutPoint)
        
        rTolThreshold = p.rTol * where(abs_nodes_infinum < abs_bestKnownObjValueWithoutPoint, abs_nodes_infinum, abs_bestKnownObjValueWithoutPoint)
        
        cond_exclude = bestKnownObjValueWithoutPoint - nodes_infinum < rTolThreshold

        l = len(where(cond_exclude)[0])
        if p.debug and l != 0: # add p.debug to onit str rendering
            p.debugmsg('num excluded by rTol: %d from %d' % (l, len(AllNodes)))
        
        cutLevel = PythonMin([cutLevel] + [AllNodes[j].key for j in where(cond_exclude)[0]])
        cond_remain = logical_not(cond_exclude)
        ind = where(cond_remain)[0]
        cutLevel = PythonMin([cutLevel] + [AllNodes[j].key for j in where(cond_exclude)[0]])
        AllNodes = [AllNodes[i] for i in ind]
    
        
        
    if p.istop != 0: 
        return AllNodes, cutLevel, threshold, None, Solutions, xRecord, bestKnownObjValueWithoutPoint, bestKnownObjValueWithPoint
    if isSNLE and maxSolutions == 1 and Min <= fTol:
        # TODO: rework it for nonlinear systems with non-bound constraints
        p.istop, p.msg = 1000, 'required solution has been obtained'
        return AllNodes, cutLevel, threshold, None, Solutions, xRecord, bestKnownObjValueWithoutPoint, bestKnownObjValueWithPoint
    
    AllNodes, cutLevel = removeSomeNodes(AllNodes, threshold, cutLevel, p)

    nCut = maxNodes#1 if fd_obj.isUncycled and all(isfinite(Lf)) and p._isOnlyBoxBounded and not p.probType.startswith('MI') else maxNodes

    AllNodes, cutLevel = TruncateOutOfAllowedNumberNodes(AllNodes, nCut, cutLevel, p)
    nNodes.append(len(AllNodes))
    
    return AllNodes, cutLevel, threshold, semiinvariant, Solutions, xRecord, bestKnownObjValueWithoutPoint, bestKnownObjValueWithPoint


def processCandidates(PointCoords, PointVals, fTol, varTols, Solutions):
    solutions, coords = Solutions.solutions, Solutions.coords
    
    candidatesInd =  where(PointVals < fTol)[0]

    candidates = PointCoords[candidatesInd]
    
    for c in candidates:
        if len(solutions) == 0 or not any(all(abs(c - coords) < varTols, axis=1)): 
            solutions.append(c)
            #coords = asarray(solutions)
            Solutions.coords = append(Solutions.coords, c.reshape(1, -1), 0)
            
    return Solutions


def getFdata(Lx, Ux, ooVars, p, fd_obj, dataType, bestKnownObjValueWithoutPoint, nlhc, semiinvariant, indT):
    
    Lx, Ux, Lf, Uf, definiteRange, exactRange, semiinvariant, indT = \
    getIntervals2(Lx, Ux, ooVars, fd_obj, dataType, p, bestKnownObjValueWithoutPoint, semiinvariant, indT)

    if Lf is None:
        return Lx, Ux, Lf, Uf, bestKnownObjValueWithoutPoint, semiinvariant, indT
    if p.debug and (Uf + 1e-15 < Lf).any():  
        p.warn('interval lower bound exceeds upper bound, it seems to be FuncDesigner kernel bug')
    if p.debug and any(logical_xor(isnan(Lf), isnan(Uf))):
        p.err('bug in FuncDesigner intervals engine')
    
    n = p.n
    m = Lf.size // (2*n)
#    print('---')
#    print(len(semiinvariant), m, Lf.shape,Lx.shape)
    Lf, Uf = Lf.reshape(2*n, m).T, Uf.reshape(2*n, m).T
#    print(len(semiinvariant), m, Lf.shape,Lx.shape)
    assert m == Lx.shape[0] == semiinvariant.size
    
    
    # TODO: check is expected speedup triggers
    

    
    if p.probType not in ('SNLE', 'NLSP') and fd_obj.isUncycled \
    and p._continuous_obj_dep_variables and exactRange:# for SNLE threshold = 0
        # TODO: 
        # handle constraints with restricted domain and matrix definiteRange
        ind = where(definiteRange)[0]
        
        if ind.size != 0:
            Lf2 = Lf[ind]
            # TODO: mb remove "nlhc is None checks", that currently remain for more safety 
            # and mb further interalg changes
            if nlhc is not None:
                nlhc = nlhc[ind]
            # TODO: if Lf has at least one -inf => prob is unbounded in the area Lx <=x <= Ux (mb inform it with certificate?)
            # TODO: is nlhc ever equals to None now? Yet it should remain for more safety, mb for future changes
            tmp1 = Lf2[nlhc==p.solver._constraintInactiveValue] if nlhc is not None else Lf2
            if tmp1.size != 0:
                tmp1 = nanmin(tmp1)
                
                ## to prevent roundoff issues ##
                tmp1 += 1e-14*abs(tmp1)
                if tmp1 == 0: tmp1 = 1e-300 
                ######################
                
                bestKnownObjValueWithoutPoint = nanmin((bestKnownObjValueWithoutPoint, tmp1)) 
        
    return Lx, Ux, Lf, Uf, bestKnownObjValueWithoutPoint, semiinvariant, indT

def updateNodes(nodesToUpdate, threshold, fTol):
    if len(nodesToUpdate) == 0: return
    Uf_tmp = vstack([node.Uf for node in nodesToUpdate])
    Tmp = Uf_tmp
    Tmp[Tmp>threshold] = threshold                

    Lf_tmp = vstack([node.Lf for node in nodesToUpdate])
    Tmp -= Lf_tmp
    
    #changes
    Tmp[Tmp<1e-300] = 1e-300
#    Tmp[Tmp<fTol] = 1e-300#fTol
    
# doesn't matter now, should be cleared in other part of interalg code      
#    Tmp[Lf_tmp>threshold] = nan
    tnlh_full_new =  - log2(Tmp)
    
    del Tmp, Uf_tmp
    
    tnlh_full_new += vstack([node.tnlh_fixed for node in nodesToUpdate])#tnlh_fixed[ind_update]
    if nodesToUpdate[0].nlhc is not None:
        nlhc = vstack([node.nlhc for node in nodesToUpdate])
        tnlh_full_new += nlhc
    
    tnlh_curr_best = nanmin(tnlh_full_new, 1)

# doesn't matter now, should be cleared in other part of interalg code  
#    Lf_tmp[Lf_tmp > threshold] = threshold + 1.0
    M = atleast_1d(nanmax(Lf_tmp, 1))
    for j, node in enumerate(nodesToUpdate): 
        node.threshold = threshold
        node.tnlh_curr = tnlh_full_new[j]
        node.tnlh_curr_best = tnlh_curr_best[j]
        node.th_key = M[j]

#    return tnlh_obj_new, tnlh_curr_best, M


#from multiprocessing import Pool
#from numpy import array_split
#def updateNodes(nodesToUpdate, threshold, p):
#    if p.nProc == 1:
#        Chunks = [nodesToUpdate]
#        result = [updateNodesEngine((nodesToUpdate, threshold))]
#    else:
#        Chunks = array_split(nodesToUpdate, p.nProc)
#        if not hasattr(p, 'pool'):
#            p.pool = Pool(processes = p.nProc)
#        #result = p.pool.imap(updateNodesEngine, [(c, threshold) for c in Chunks])
#        result = p.pool.map(updateNodesEngine, [(c, threshold) for c in Chunks])
#    for i, elem in enumerate(result):
#        if elem is None: continue
#        tnlh_all_new, tnlh_curr_best, M = elem
#        for j, node in enumerate(Chunks[i]): 
#            node.threshold = threshold
#            node.tnlh_curr = tnlh_all_new[j]
#            node.tnlh_curr_best = tnlh_curr_best[j]
#            node.th_key = M[j]
