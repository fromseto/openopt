PythonSum = sum
from numpy import isnan, array, atleast_1d, asarray, logical_and, all, logical_or, any, \
arange, vstack, inf, logical_not, take, abs, empty, \
isfinite, argsort, ones, zeros, log1p, array_split#where

# for PyPy
from openopt.kernel.nonOptMisc import where

#try:
#    from bottleneck import nanargmin, nanmin
#except ImportError:
#    from numpy import nanmin, nanargmin
from interalgLLR import *

def getMOPnlh_seq(Arg):
    targets_vals, targets_tols, solutionsF, lf, uf = Arg
    lf, uf = asarray(lf), asarray(uf)
    if lf.size == 0 or len(solutionsF) == 0: 
        return None

    m = lf.shape[0]
    n = lf.shape[2]//2
    r = zeros((m, 2*n))

    for _s in solutionsF:
        s = atleast_1d(_s)
        tmp = ones((m, 2*n))
        for i in range(len(targets_vals)):
            val, tol = targets_vals[i], targets_tols[i]#, t.val, t.tol
            #TODO: mb optimize it
            Lf, Uf = lf[:, i], uf[:, i] 
            if val == inf:
                ff = s[i] + tol
                ind = Uf > ff
                if any(ind):
                    t1 = Uf[ind]
                    t2 = Lf[ind]
                    t_diff = t1-t2
                    t_diff[t_diff<1e-200] = 1e-200
                    # TODO: check discrete cases
                    Tmp = (ff-t2) / t_diff

                    tmp[ind] *= Tmp
                    tmp[ff<Lf] = 0.0
                    
            elif val == -inf:
                ff = s[i] - tol
                ind = Lf < ff
                if any(ind):
                    t1 = Uf[ind]
                    t2 = Lf[ind]
                    t_diff = t1-t2
                    t_diff[t_diff<1e-200] = 1e-200
                    
                    Tmp = (t1-ff) / t_diff

                    tmp[ind] *= Tmp
                    tmp[Uf<ff] = 0.0
            else: # finite val
                ff = abs(s[i]-val) - tol
                if ff <= 0:
                    continue
                _lf, _uf = Lf - val, Uf - val
                ind = logical_or(_lf < ff, _uf > - ff)
                _lf = _lf[ind]
                _uf = _uf[ind]
                _lf[_lf>ff] = ff
                _lf[_lf<-ff] = -ff
                _uf[_uf<-ff] = -ff
                _uf[_uf>ff] = ff
                
                Diff = Uf[ind] - Lf[ind]
                Diff[Diff<1e-200] = 1e-200
                _diff = _uf - _lf
                _diff[_diff<1e-200] = 1e-200
                
                Tmp = 1.0 - (_uf - _lf) / Diff
                
                tmp[ind] *= Tmp

        new = 0
        if new:
            ind_0 = tmp == 0.0
            ind_1 = tmp == 1.0
            r[ind_1] = inf
            ind_m = logical_not(logical_and(ind_0, ind_1))
            #r[ind_m] -= log1p(-tmp[ind_m]) * 1.4426950408889634
            r[ind_m] -= log1p(-tmp[ind_m]) * 1.4426950408889634
            #Diff = log2(1-tmp[ind_m]) #* 1.4426950408889634
            
        else:
            r -= log1p(-tmp) * 1.4426950408889634 # log2(e)
            
        #r -= Diff
    return r

from multiprocessing import Pool
def getMOPnlh(targets, SolutionsF, lf, uf, pool, nProc):
    lf, uf = asarray(lf), asarray(uf)
    target_vals = [t.val for t in targets]
    target_tols = [t.tol for t in targets]
    if nProc == 1 or len(SolutionsF) <= 1:
        return getMOPnlh_seq((target_vals, target_tols, SolutionsF, lf, uf))
        
    splitBySolutions = True #if len(SolutionsF) > max((4*nProc, ))
    if splitBySolutions:
        ss = array_split(SolutionsF, nProc)
        Args = [(target_vals, target_tols, s, lf, uf) for s in ss]
        result = pool.imap_unordered(getMOPnlh_seq, Args)#, callback = cb)    
        r = [elem for elem in result if elem is not None]
        return PythonSum(r)
    else:
        lf2 = array_split(lf, nProc)
        uf2 = array_split(uf, nProc)
        Args = [(target_vals, target_tols, SolutionsF, lf2[i], uf2[i]) for i in range(nProc)]
        result = pool.map(getMOPnlh_seq, Args)
        r = [elem for elem in result if elem is not None]
        return vstack(r)

def processBoxesMOP(p, nlhc, residual, definiteRange, Lx, Ux, ooVars, fd_obj, C, bestKnownObjValueWithPoint, cutLevel, nNodes,  \
         bestKnownObjValueWithoutPoint, fTol, Solutions, varTols, inactiveNodes, dataType, \
         maxNodes, semiinvariant, indTC, xRecord, activeCons):

    assert p.probType == 'MOP'
    
    fd_obj = [t.func for t in p.targets]
    
#    if len(p._discreteVarsNumList):
#        Lx, Ux, ind_trunc = adjustDiscreteVarBounds(Lx, Ux, p)
#        if ind_trunc.size:
#            
#            semiinvariant, indTC, 
    
    
    if p.nProc != 1 and getattr(p, 'pool', None) is None:
        p.pool = Pool(processes = p.nProc)
    elif p.nProc == 1:
        p.pool = None
    
    Lfl, Ufl = [], []
    targets = p.targets # TODO: check it
    m, n = Lx.shape
    Lfl, Ufl = [[] for k in range(m)], [[] for k in range(m)]
    ind_nan = np.zeros(m, bool)
    for i, t in enumerate(targets):
        Lx, Ux, Lf, Uf, definiteRange, exactRange, semiinvariant_, indTC_ = getIntervals2(Lx, Ux, ooVars, t.func, dataType, p)
#        Lf, Uf, definiteRange, exactRange, semiinvariant, indTC = getIntervals2(Lx, Ux, ooVars, t.func, dataType, p, semiinvariant, indTC)
        Lf, Uf = Lf.reshape(2*n, m).T, Uf.reshape(2*n, m).T
        ind_nan |= logical_and(all(isnan(Lf), axis=1), np.all(isnan(Uf), axis=1))
        for j in range(m):
            Lfl[j].append(Lf[j])
            Ufl[j].append(Uf[j])
        #Lfl.append(Lf.reshape(2*n, m).T.tolist())
        #Ufl.append(Uf.reshape(2*n, m).T.tolist())

    threshold_prev = 0
    
    if Lx.size == 0:
        return inactiveNodes, cutLevel, threshold_prev, semiinvariant, Solutions, xRecord, bestKnownObjValueWithoutPoint, bestKnownObjValueWithPoint
    
    nodes, Lx, Ux, Lf, Uf, semiinvariant, indT, nlhc, residual, activeCons = formNodes(Lx, Ux, nlhc, indTC, residual, Lfl, Ufl, semiinvariant, p, activeCons, ind_nan)
    if len(nodes) == 0: # after remove some nodes with NaNs
        return inactiveNodes, cutLevel, threshold_prev, semiinvariant, Solutions, xRecord, bestKnownObjValueWithoutPoint, bestKnownObjValueWithPoint
        
    m, n = Lx.shape # updated value
    assert semiinvariant.size == m == len(Lfl) == Lx.shape[0]
    
    nlh_obj = getMOPnlh(targets, Solutions.F, Lfl, Ufl, p.pool, p.nProc)
    
    #Lx, Ux = TruncateSomeBoxes(Lx, Ux, Lf, Uf, threshold)
    
    
    assert p.solver.dataHandling == 'raw', '"sorted" mode is unimplemented for MOP yet'
    
    if nlh_obj is None:
        new_nodes_tnlh_all = nlhc
    elif nlhc is None: 
        new_nodes_tnlh_all = nlh_obj
    else:
        new_nodes_tnlh_all = nlh_obj + nlhc

    candidatesF, candidatesCoords = getCentersValues(ooVars, Lx, Ux, activeCons, new_nodes_tnlh_all, fd_obj, C, p.contol, dataType, p) 

    threshold = 0 # unused for MOP
    
    nIncome, nOutcome = updateWPF(Solutions, candidatesCoords, candidatesF, targets, p.solver.sigma)
    
    p._frontLength = len(Solutions.F)
    p._nIncome = nIncome
    p._nOutcome = nOutcome
    p.iterfcn(p.x0)
    #print('iter: %d (%d) frontLenght: %d' %(p.iter, itn, len(Solutions.coords)))


    # TODO: better of nlhc for unconstrained probs

#    if len(inactiveNodes) != 0:
#        AllNodes = hstack((nodes,  inactiveNodes))
#    else:
#        AllNodes = atleast_1d(nodes)

    AllNodes = nodes + inactiveNodes

    
    if p.istop != 0: 
        return AllNodes, cutLevel, threshold, None, Solutions, xRecord, bestKnownObjValueWithoutPoint, bestKnownObjValueWithPoint
        
    

    

    hasNewParetoNodes = False if nIncome == 0 else True
    if hasNewParetoNodes:
        nodesForRecalculation = nodes + inactiveNodes
        Lfl2 = [node.Lf for node in nodesForRecalculation]
        Ufl2 = [node.Uf for node in nodesForRecalculation]
        nlhc2 = [node.nlhc for node in nodesForRecalculation]
        nlh_obj2 = getMOPnlh(targets, Solutions.F, Lfl2, Ufl2, p.pool, p.nProc)
        tnlh_all = asarray(nlhc2) if nlh_obj2 is None else nlh_obj2 if nlhc2[0] is None else asarray(nlhc2) + nlh_obj2
        
        IND = logical_not(any(isfinite(tnlh_all), 1))
        if any(IND):
            ind = where(logical_not(IND))[0]
            AllNodes = [AllNodes[i] for i in ind]
            tnlh_all = take(tnlh_all, ind, axis=0, out=tnlh_all[:ind.size])        
        work_elems = (AllNodes, tnlh_all)
    else:
        work_elems = (nodes, new_nodes_tnlh_all)
    
    Nodes, Tnlh_all = work_elems
    IND = logical_not(any(isfinite(Tnlh_all), 1))
    if any(IND):
        ind = where(logical_not(IND))[0]
        for i in where(IND)[0][::-1]:
            Nodes.pop(i)
        if hasNewParetoNodes:
            Tnlh_all = tnlh_all = take(tnlh_all, ind, axis=0, out=tnlh_all[:ind.size])
            AllNodes = nodes + inactiveNodes 
        else:
            Tnlh_all = new_nodes_tnlh_all = take(new_nodes_tnlh_all, ind, axis=0, out=new_nodes_tnlh_all[:ind.size])
    
    if len(AllNodes) == 0:
        return inactiveNodes, cutLevel, threshold_prev, semiinvariant, Solutions, xRecord, bestKnownObjValueWithoutPoint, bestKnownObjValueWithPoint
        
    

    n = Tnlh_all.shape[1] // 2
    T1, T2 = Tnlh_all[:, :n], Tnlh_all[:, n:]
    T = where(logical_or(T1 < T2, isnan(T2)), T1, T2)
    bestCoordsForSplitting = asarray(nanargmin(T, 1), int)
    
    arangeM = arange(bestCoordsForSplitting.size)
    NN = T[arangeM, bestCoordsForSplitting].flatten()# TODO: IS flatten() required here?

    for i, node in enumerate(Nodes):
        # low priority, mb rework/remove it
        node.tnlh_curr_best = NN[i]# used only for estimation of number of nodes to involve
        
    if p.solver.mop_mode == 1:
        for i, node in enumerate(Nodes):
            node.tnlh_all = Tnlh_all[i] # TODO: rework/remove it
    elif len(nodes) != 0:
        Lxc, Uxc = vstack([n.Lx for n in nodes]), vstack([n.Ux for n in nodes])
        Lfc, Ufc = vstack([n.Lf for n in nodes]), vstack([n.Uf for n in nodes])
        
        # !!!!!!! TODO: implement Lf, Uf truncution for MOP there !!!!!!
        indT = TruncateSomeBoxes(p, Lxc, Uxc, Lfc, Ufc, threshold, new_nodes_tnlh_all)
        
        _indT = array([n.indtc for n in nodes])
        indT &= _indT
        for i, n in enumerate(nodes):
            n.indtc = indT[i]
        
        

    astnlh = argsort(NN)

    if hasNewParetoNodes:
        AllNodes = [AllNodes[i] for i in astnlh]
        p._bestCoordsForSplitting = bestCoordsForSplitting
    else:# Nodes is nodes
        nodes = [nodes[i] for i in astnlh]
        AllNodes = nodes + inactiveNodes
        tmp = getattr(p, '_bestCoordsForSplitting', [])
       
        p._bestCoordsForSplitting = np.hstack((bestCoordsForSplitting, tmp)) if len(tmp) else bestCoordsForSplitting

    
    
    #assert semiinvariant.size == m == len(Lfl) == Lx.shape[0]
    
    
    # TODO: form semiinvariant in other level (for active nodes only), to reduce calculations
#    if len(AllNodes) != 0:
#        T = asarray([node.nlh_obj_fixed for node in AllNodes])
##        nlhc_fixed = asarray([node.nlhc for node in AllNodes])
#        if AllNodes[0].nlhc is not None:
#            T += asarray([node.nlhc for node in AllNodes])
##        T = nlhf_fixed + nlhc_fixed if nlhc_fixed[0] is not None else nlhf_fixed 
#        p._semiinvariant = \
#        nanmin(vstack(([T[arangeM, bestCoordsForSplitting], T[arangeM, n+bestCoordsForSplitting]])), 0)
#    else:
#        p._semiinvariant = array([])

#        p._nObtainedSolutions = len(solutions)
#        if p._nObtainedSolutions > maxSolutions:
#            solutions = solutions[:maxSolutions]
#            p.istop = 0
#            p.msg = 'user-defined maximal number of solutions (p.maxSolutions = %d) has been exeeded' % p.maxSolutions
#            return AllNodes, cutLevel, threshold, None, solutions, coords, xRecord, bestKnownObjValueWithoutPoint, bestKnownObjValueWithPoint
    
    

    #AllNodes, cutLevel = removeSomeNodes(AllNodes, threshold, cutLevel, p)

    nCut = maxNodes#1 if fd_obj.isUncycled and all(isfinite(Lf)) and p._isOnlyBoxBounded and not p.probType.startswith('MI') else maxNodes
    
    AllNodes, cutLevel = TruncateOutOfAllowedNumberNodes(AllNodes, nCut, cutLevel, p)
    nNodes.append(len(AllNodes))
    
    assert semiinvariant.size == m == len(Lfl) == Lx.shape[0]
    
    return AllNodes, cutLevel, threshold, semiinvariant, Solutions, xRecord, bestKnownObjValueWithoutPoint, bestKnownObjValueWithPoint



def updateWPF(Solutions, candidatesCoords, candidatesF, targets, sigma):
#    print Solutions.F
#    if len(Solutions.F) != Solutions.coords.shape[0]:
#        raise 0
    # TODO: rework it
    #sf = asarray(Solutions.F)
    nIncome, nOutcome = 0, 0
    m= len(candidatesCoords)
    #n = len(candidatesCoords[0])
    # TODO: mb use inplace candidatesCoords / candidatesF modification instead?
    for j in range(m):
        if np.any(isnan(candidatesF[j])):
            continue
        if Solutions.coords.size == 0:
            Solutions.coords = array(candidatesCoords[j]).reshape(1, -1)
            Solutions.F.append(candidatesF[0])
            nIncome += 1
            continue
        M = Solutions.coords.shape[0] 
        
        Candidat_better = empty(M, bool)
        Candidat_better.fill(False)
#        Solution_better = empty(M, bool)
#        Solution_better.fill(False)
        for i, target in enumerate(targets):
            
            f = candidatesF[j][i]
            
            # TODO: rewrite it
            F = asarray([Solutions.F[k][i] for k in range(M)])
            #d = f - F # vector-matrix
            
            val, tol = target.val, target.tol
            Tol = sigma * tol
            if val == inf:
                ind_candidat_better = f > F + Tol
#                ind_solution_better = f <= F#-tol
            elif val == -inf:
                ind_candidat_better = f < F - Tol
#                ind_solution_better = f >= F#tol
            else:
                ind_candidat_better = abs(f - val) < abs(F - val) - Tol
#                ind_solution_better = abs(f - val) >= abs(Solutions.F[i] - val)#-tol # abs(Solutions.F[i] - target)  < abs(f[i] - target) + tol
            
            Candidat_better = logical_or(Candidat_better, ind_candidat_better)
#            Solution_better = logical_or(Solution_better, ind_solution_better)
        
        accept_c = all(Candidat_better)
        #print sum(asarray(Solutions.F))/asarray(Solutions.F).size
        if accept_c:
            nIncome += 1
            #new
            Solution_better = empty(M, bool)
            Solution_better.fill(False)
            for i, target in enumerate(targets):
                f = candidatesF[j][i]
                F = asarray([Solutions.F[k][i] for k in range(M)])
                val, tol = target.val, target.tol
                if val == inf:
                    ind_solution_better = f < F
                elif val == -inf:
                    ind_solution_better = f > F
                else:
                    ind_solution_better = abs(f - val) > abs(F - val)
                Solution_better = logical_or(Solution_better, ind_solution_better)

            SolutionsToBeRemoved = logical_not(Solution_better)
            remove_s = any(SolutionsToBeRemoved)
            if remove_s:
                indRemove = where(SolutionsToBeRemoved)[0]
                nOutcome += indRemove.size
                Solutions.coords[indRemove[0]] = candidatesCoords[j]
                Solutions.F[indRemove[0]] = candidatesF[j]
                
                if indRemove.size > 1:
                    SolutionsToBeRemoved[indRemove[0]] = False
                    indLeft = logical_not(SolutionsToBeRemoved)
                    indLeftPositions = where(indLeft)[0]
                    newSolNumber = Solutions.coords.shape[0] - indRemove.size + 1
                    Solutions.coords = take(Solutions.coords, indLeftPositions, axis=0, out = Solutions.coords[:newSolNumber])
                    Solutions.F = [Solutions.F[i] for i in indLeftPositions]
            else:
                Solutions.coords = vstack((Solutions.coords, candidatesCoords[j]))
                Solutions.F.append(candidatesF[j])
    return nIncome, nOutcome
