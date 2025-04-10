PythonSum = sum
PythonAll = all
from numpy import empty, logical_and, logical_not, take, zeros, isfinite, any, \
asarray, ndarray #where
from interalgT import truncateByPlane, getTruncatedArrays, splitDomainForDiscreteVariable#adjustDiscreteVarBounds
import numpy as np
from bisect import bisect_left


# for PyPy
from openopt.kernel.nonOptMisc import where

hasPoint = lambda Lx, Ux, point:\
    True if Lx.size != 0 and any([(np.all(Lx[i]<=point) and np.all(Ux[i]>=point)) for i in range(Lx.shape[0])]) else False
pointInd = lambda Lx, Ux, point:\
    where([(np.all(Lx[i]<=point) and np.all(Ux[i]>=point)) for i in range(Lx.shape[0])])[0].tolist()

a = np.array([  2.61799322e+00,   -1.28399661e+00,    1.05879069e+00,   -2.81291365e-07])
    
def processConstraints(C, C0, Lx, Ux, semiinvariant, inactiveNodes, p, dataType, activeCons):
    assert len(activeCons) == semiinvariant.size
    #P = np.array([  7.64673334e-01,    4.35551807e-01,    5.93869991e+02,   5.00000000e+00])
#    P = np.array([-0.63521194458007812, -0.3106536865234375, 0.0905609130859375, 0.001522064208984375, -0.69999999999999996, -0.99993896484375, 0.90000152587890625, 1.0, 4.0])

#    print('c-1', p.iter, hasPoint(Lx, Ux, P), pointInd(Lx, Ux, P))

#    allNodesAreActive = len(inactiveNodes)==0
    n = p.n
    m = Lx.shape[0]
#    print('iter:',p.iter, 'm>>>:', m, p._constraints_counter)
    indT = empty(m, bool)
    indT.fill(False)
#    isSNLE = p.probType in ('NLSP', 'SNLE')
    
    
    # TODO: involve reducing of linear constraints
    
#    for i in range(p.nb):
#        Lx, Ux, indT, ind_remain = truncateByPlane(Lx, Ux, indT, p.A2[i], p.b2[i]+p.contol)
#        if ind_remain is not None:
#            semiinvariant = semiinvariant[ind_remain]
#            activeCons = [activeCons[j] for j in ind_remain]
    
    lin_ineq_ids = p._lin_ineq_ids
    lin_eq_ids = p._lin_eq_ids
#    print('>>>>>>', len(lin_ineq_ids))
    
    
    if 1:
        for line_A, val_b in lin_ineq_ids.values():
            Lx, Ux, indT, ind_remain = truncateByPlane(Lx, Ux, indT, line_A, val_b + p.contol)
            if ind_remain is not None:
                semiinvariant = semiinvariant[ind_remain]
                activeCons = [activeCons[j] for j in ind_remain]
        
        

        for line_A, val_b in lin_eq_ids.values():
            # TODO: handle it via one func
            Lx, Ux, indT, ind_remain = truncateByPlane(Lx, Ux, indT, line_A, val_b+p.contol)
            if ind_remain is not None:
                semiinvariant = semiinvariant[ind_remain]
                activeCons = [activeCons[j] for j in ind_remain]
            Lx, Ux, indT, ind_remain = truncateByPlane(Lx, Ux, indT, -line_A, -val_b+p.contol)
            if ind_remain is not None:
                semiinvariant = semiinvariant[ind_remain]
                activeCons = [activeCons[j] for j in ind_remain]
    
    DefiniteRange = True
    m = Lx.shape[0]
    nlh = zeros((m, 2*n))
    nlh_0 = zeros(m)

    # TODO: probably remove it
    fullOutput = False#isSNLE and not p.hasLogicalConstraints
    
    residual = None#zeros((m, 2*n)) if fullOutput else None
#    residual_0 = zeros(m) if fullOutput else None
    
#    inactiveConsIdx = []
    ind_discrete = np.array(p._discreteVarsNumList, int)# TODO: omit recreating it each time
    
    newActiveCons = [[] for j in range(m)]
#    newActiveCons = [[]]*m# for j in range(m)]

    
    
    constraints_counter = p._constraints_counter
#    print('-----------')
#    print('len(inactiveNodes):', len(inactiveNodes), 'm:', m)
#    print('constraints_counter:', constraints_counter)
    freeVarsDict = p._freeVarsDict
    freeVarsList = p._freeVarsList
    
    
    for c_id, (c, f, lb, ub, tol) in C0.items():
#        print ('c_1', p.iter, c.dep, hasPoint(Lx, Ux, a))
        m = Lx.shape[0] # is changed in the cycle

        if m == 0: 
            return Lx.reshape(0, n), Ux.reshape(0, n), nlh.reshape(0, 2*n), \
            residual, True, False, semiinvariant, []

        # >>>>>>>>> NEW
        # form Lx, Ux wrt active constraints
        active_points_inds_list = []
        for j in range(m):
            
            boxActiveConstrList = activeCons[j]
            
# This check eats too much cputime
#            assert PythonAll(x <= y for x,y in zip(boxActiveConstrList, boxActiveConstrList[1:])) 

            tmp = bisect_left(boxActiveConstrList, c_id)
            
            if tmp != len(boxActiveConstrList) and boxActiveConstrList[tmp] == c_id:
                active_points_inds_list.append(j)
                
#        active_points_inds_list = [j for j in range(m) if ActiveCons[j][bisect_left(ActiveCons[j], c_id)] == c_id]
        nActivePoints = len(active_points_inds_list)
        
        if p.solver._constraints_reduction:
            if nActivePoints == 0:
#                print('case 1', c_id)
                continue
            elif nActivePoints == m:
#                print('case 2', c_id)
                useReduction = False
                Slice = slice(None)
                # reducedLx, reducedUx
                rLx, rUx = Lx, Ux
            else:
#                print('case 3', c_id)
                useReduction = True
                active_points_inds = np.array(active_points_inds_list, int)
                
                if 1:
                    Slice = active_points_inds
                else:
                    Slice = slice(None)
                    useReduction = False
                    
                # reducedLx, reducedUx
                rLx, rUx = Lx[Slice], Ux[Slice]
        else:
            useReduction = False
            Slice = slice(None)
            # reducedLx, reducedUx
            rLx, rUx = Lx, Ux


        if fullOutput:
            assert 0, 'unimplemented'
            # TODO: add active_points_inds here
#                (T0, Res0), (res, R_res), DefiniteRange2 = c.nlh(rLx, rUx, p, dataType, fullOutput = True)
#                residual_0 += Res0
        else:
            # may be logical constraint and doesn't have kw fullOutput at all
            T0, res, DefiniteRange2 = c.nlh(rLx, rUx, p, dataType)
            
        if not np.all(DefiniteRange2):
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!TODO: check this code!
            if not useReduction:
                DefiniteRange = logical_and(DefiniteRange, DefiniteRange2)
            else:
                DefiniteRange_ = np.empty(m, bool)
                DefiniteRange_.fill(True) 
                DefiniteRange_[Slice] = DefiniteRange2
                DefiniteRange = logical_and(DefiniteRange, DefiniteRange_)
        
        activeConstraintBoxes = where((T0 != 0) | logical_not(DefiniteRange2))[0]
        
        if activeConstraintBoxes.size == 0:
            continue # the cycle by constraints


        for i in activeConstraintBoxes:
            newActiveCons[active_points_inds[i] if useReduction else i].append(c_id)
        
        
        assert T0.ndim <= 1, 'unimplemented yet'
        
        
        nlh_0[Slice] += T0
#        if Slice != slice(None):
#            nlh_0[Slice] += T0
#        else:
#            nlh_0 += T0
        
        
#        if useReduction:
#            
#        else:
#            nlh_0 += T0
#        assert nlh.shape[0] == m
        # TODO: rework it for case len(p._freeVarsList) >> 1
        if p.solver.prioritized_constraints:
            AC_lst = [activeCons[k] for k in active_points_inds] if useReduction else activeCons
            c_nlh_multiplier = getMultiplier(AC_lst, c_id, p.iter)
            
        for v, tmp in res.items():
            j = freeVarsDict.get(v)
            half_width = tmp.shape[1]//2
            a1, a2 = tmp[:, half_width:].flatten() - T0, tmp[:, :half_width].flatten() - T0
            if p.solver.prioritized_constraints:
                a1 *= c_nlh_multiplier
                a2 *= c_nlh_multiplier
            nlh[Slice, n+j] += a1
            nlh[Slice, j] += a2

            if fullOutput:
                assert 0, 'unimplemented'
#                Tmp = R_res[v]
#                residual[:, n+j] += Tmp[:, Tmp.shape[1]//2:].flatten() - Res0
#                residual[:, j] += Tmp[:, :Tmp.shape[1]//2].flatten() - Res0

#        assert nlh.shape[0] == m

        # TODO: mb operate on nlh[Slice] instead of full nlh here to speedup computations
        
        # TODO: mb use it earlier before nlh update computations with tmp - T0
        ind_remain = logical_and(any(isfinite(nlh), axis=1), isfinite(nlh_0))
        ind_remain |= logical_not(DefiniteRange)
        Ind_remain = where(ind_remain)[0]

        
        lj = Ind_remain.size
        if lj != m:
#            print('removing: ', lj)
#            assert 0
#            assert nlh.shape[0] == Lx.shape[0]
            Lx = take(Lx, Ind_remain, axis=0, out=Lx[:lj])
            Ux = take(Ux, Ind_remain, axis=0, out=Ux[:lj])
            nlh = take(nlh, Ind_remain, axis=0, out=nlh[:lj])
            nlh_0 = nlh_0[Ind_remain]
#            T0 = T0[ind]

#            residual = take(residual, ind, axis=0, out=residual[:lj])
            indT = indT[Ind_remain]
            semiinvariant = semiinvariant[Ind_remain]
            activeCons = [activeCons[i] for i in Ind_remain]
            newActiveCons = [newActiveCons[i] for i in Ind_remain]
            if fullOutput:
                assert 0, 'unimplemented'
#                residual_0 = residual_0[ind]
#                residual = take(residual, ind, axis=0, out=residual[:lj])
            if asarray(DefiniteRange).size != 1: 
                DefiniteRange = take(DefiniteRange, Ind_remain, axis=0, out=DefiniteRange[:lj])
#            print ('c_2', p.iter, c.dep, hasPoint(Lx, Ux, a))
#        assert nlh.shape[0] == Lx.shape[0]

        # filter nlh from infs and nans
        # TODO: mb operate on nlh[Slice] instead of full nlh here to speedup computations
        ind = logical_not(isfinite(nlh)) # 2D matrix
        #ind_int = where(ind) # 2D matrix
        
        if any(ind):
#            Lx, Ux, nlh, indT = processHalving_prev(p, ind, Lx, Ux, nlh, indT, ind_discrete, freeVarsList, fullOutput)
            s1 = Lx.shape
            Lx, Ux, nlh, indT = processHalving(ind, Lx, Ux, nlh, indT, ind_discrete, freeVarsList, fullOutput)
            s2 = Lx.shape
            assert s1 == s2
            
            
        # changes

#        if p.solver._constraints_reduction:
#            for elem in newActiveCons:
#                if len(elem)!=0 and elem[-1] == c_id:
#                    constraints_counter[c_id] += 1 
            #constraints_counter[c_id] += (T0 != 0).sum()#asarray(T0 != 0, np.int8).sum()
            #############

        '''                  End of cycle by constraints                  '''


    if p.solver._constraints_reduction:
        for box_active_constraints in newActiveCons:
            for c_id in box_active_constraints:
#            if len(elem)!=0 and elem[-1] == c_id:
                constraints_counter[c_id] += 1 
        
    if not all(v >= 0 for v in constraints_counter.values()):
        print('negative constr counter at iter %d:'  % p.iter)
        print('\t',[(k,v) for k,v in constraints_counter.items() if v < 0])

    
#    if p.iter >= 10:
#        print('9:', c_id, hasPoint(Lx, Ux, a))
    

    '                   Updating list of active constraints                   '
#    print(constraints_counter)
    if p.solver._constraints_reduction:
#        print('iter:',p.iter, '>>>:', p._constraints_counter)
        new_active_cons = set()
        new_active_cons.update(*newActiveCons)
        
        storedInactiveCons = set(_id for _id, num in constraints_counter.items() if num == 0)
        
        inactiveConsIdx = storedInactiveCons - new_active_cons
        
        if len(inactiveConsIdx) != 0:
            if p.iprint >= 0:
                l = len(inactiveConsIdx)
                s = 'removing %d inactive constraint%s\t| %d remain%s' % \
                (l,'' if l==1 else 's', len(C)-l, 's' if len(C)-l==1 else '')
                if p.iter == 1:
                    s += ' in interalg initialization phase'
                p.disp(s)
            for j in inactiveConsIdx:
    #            print('removing', C[j][0].expr)
                C.pop(j), C0.pop(j), constraints_counter.pop(j)
        
            for elem in inactiveConsIdx:
                if lin_ineq_ids.pop(elem, None) is not None: continue
                lin_eq_ids.pop(elem, None)   



    '                              Updating nlh                              '
    if nlh.size != 0:
        if DefiniteRange is False:
            nlh_0 += 1e-300
        elif type(DefiniteRange) == ndarray and not np.all(DefiniteRange):
            nlh_0[logical_not(DefiniteRange)] += 1e-300
    # !! matrix - vector operation !!
    nlh += nlh_0.reshape(-1, 1)
    
    if fullOutput:
        assert 0, 'unimplemented'
        # !! matrix - vector
#        residual += residual_0.reshape(-1, 1)
#        residual[residual_0>=1e300] = 1e300
    
    #print('c2', p.iter, hasPoint(Lx, Ux, P), pointInd(Lx, Ux, P))
#    print('newActiveCons:', newActiveCons)
#    print('-----------------')
#    print(newActiveCons)

#    print('end of interalg_cons: cons_counter = ', constraints_counter)

    assert len(newActiveCons) == len(activeCons)
    

    if p.solver.prioritized_constraints:
        ac2 = []
        from interalg_oo import ActiveConstraintsEntry
        for i, item in enumerate(newActiveCons):
            tmp = ActiveConstraintsEntry(item)
            Tmp = activeCons[i].triggering_info
            for j in inactiveConsIdx:
                Tmp.pop(j, None)
            tmp.triggering_info = Tmp
            
            ac2.append(tmp)
            
        newActiveCons = ac2

    return Lx, Ux, nlh, residual, DefiniteRange, indT, semiinvariant, newActiveCons
  

def processHalving(ind, Lx, Ux, nlh, indT, ind_discrete, freeVarsList, fullOutput):            
    
    ind_ = any(ind, 1)
    ind_truncate = where(ind_)[0]
    indT[ind_truncate] = True
    
    ind_l,  ind_u = ind[:, :ind.shape[1]//2], ind[:, ind.shape[1]//2:]
    Ind_l,  Ind_u = where(ind_l), where(ind_u)
    tmp_l, tmp_u = 0.5 * (Lx[Ind_l] + Ux[Ind_l]), 0.5 * (Lx[Ind_u] + Ux[Ind_u])
    
    
    # 1. A block for discrete variables processing
    if ind_discrete.size != 0:
        IND_discrete = ind_discrete.reshape(-1, 1)
        #L, U = Lx[:, IND].copy(), Ux[:, IND].copy()
        L, U = Lx[ind_truncate, IND_discrete].copy(), Ux[ind_truncate, IND_discrete].copy()
    
    # 2. The line is for all variables - continuous & discrete
    Lx[Ind_l], Ux[Ind_u] = tmp_l, tmp_u
    
    
    # 3. Started another block for discrete variables        
    
    if ind_discrete.size != 0:
        #Lx[:, IND], Ux[:, IND] = L, U
        Lx[ind_truncate, IND_discrete], Ux[ind_truncate, IND_discrete] = L, U
    
    #new2 (unimplemented yet)
#            for j in ind_truncate:
##                t_idx = ind[j]
##                sz = t_idx.size
#                ind_dl, ind_du = ind_l[j], ind_u[j]
#                Ind = 
#                ind_l, ind_u = t_idx[:sz//2], t_idx[sz//2:]
#                ind_l1 = where(ind_l[, ])[0]


    # TODO: rework it (low priority), get inds of discrete variables from c.dep
    # or a new c field that handles inds of discrete variables that c depends on
    # mb other improvements
    new = 0
    if new:
        #new (unimplemented properly yet)
#                D = dict()
        ind_l_d = ind_l[:, ind_discrete]
        Ind = where(ind_l_d)
        for k, I in zip(Ind[0], Ind[1]):
            i = ind_discrete[I]
#                    tmp = D.get(i, None)
#                    if tmp is None:
#                        D[i] = [k]
#                    else:
#                        D[i].append(k)
            v = freeVarsList[i]
            mid1, mid2 = splitDomainForDiscreteVariable(
            Lx[k, i], Ux[k, i], v)
            #Lx[ind_l_d[:, k], i], Ux[ind_l_d[:, k], i], v)
            
            Lx[k, i] = mid2

        ind_l_u = ind_u[:, ind_discrete]
        Ind = where(ind_l_u)
        for k, I in zip(Ind[0], Ind[1]):
            i = ind_discrete[I]
            v = freeVarsList[i]
            mid1, mid2 = splitDomainForDiscreteVariable(
            Lx[k, i], Ux[k, i], v)
            #Lx[ind_l_u[:, k], i], Ux[ind_l_u[:, k], i], v)
            
            #Ux[ind_l_u, i] = mid1
            Ux[k, i] = mid1

    else:
        #prev
        for i in ind_discrete:
#                    if ind_truncate[i]
            ind_l1 = where(ind_l[:, i])[0]
            ind_u1 = where(ind_u[:, i])[0]
            if ind_l1.size == 0 and ind_u1.size == 0: 
                continue
#                    if not ind[:, i].any():
#                        continue
            v = freeVarsList[i]
            
            if ind_l1.size:
                mid1, mid2 = splitDomainForDiscreteVariable(Lx[ind_l1, i], Ux[ind_l1, i], v)
                Lx[ind_l1, i] = mid2
                
            
            if ind_u1.size:
                mid1, mid2 = splitDomainForDiscreteVariable(Lx[ind_u1, i], Ux[ind_u1, i], v)
                Ux[ind_u1, i] = mid1
#                    Ux[ind_u1, n+i] = U[ind_u1, i]
#                    Ux[ind_u1, i] = mid1

    '           Finished handling Lx & Ux for discrete variables           '
    


    # TODO: mb lock is required for parallel computations
    
    nlh_l, nlh_u = nlh[:, nlh.shape[1]//2:], nlh[:, :nlh.shape[1]//2]
        
    # inplace operations are performed in the cycle
    if 1:
        # TODO: mb improve it using inplace ([:])
        nlh_l[ind_u], nlh_u[ind_l] = nlh_u[ind_u].copy(), nlh_l[ind_l].copy()   
    else:
        ind_Tmp = logical_and(ind_u, logical_not(ind_l))
        nlh_l[ind_Tmp] = nlh_u[ind_Tmp].copy()
        ind_Tmp = logical_and(ind_l, logical_not(ind_u))
        nlh_u[ind_Tmp] = nlh_l[ind_Tmp].copy()
    
    if fullOutput:
        assert 0, 'unimplemented'
#                residual_l, residual_u = residual[:, residual.shape[1]//2:], residual[:, :residual.shape[1]//2]
#                residual_l[ind_u], residual_u[ind_l] = residual_u[ind_u].copy(), residual_l[ind_l].copy()   
#            print ('c_3', itn, c.dep, hasPoint(Lx, Ux, P))
   
    return Lx, Ux, nlh, indT

def processHalving_prev(p, ind, Lx, Ux, nlh, indT, ind_discrete, freeVarsList, fullOutput):            
    indT[any(ind, 1)] = True
    
    ind_l,  ind_u = ind[:, :ind.shape[1]//2], ind[:, ind.shape[1]//2:]
#            ind_ = logical_or(logical_not(ind_l), logical_not(ind_u))
    tmp_l, tmp_u = 0.5 * (Lx[ind_l] + Ux[ind_l]), 0.5 * (Lx[ind_u] + Ux[ind_u])
    
    # TODO: improve it, don't copy for continuous variables
    if len(p._discreteVarsNumList):
        IND = np.array(p._discreteVarsNumList, int)
        L, U = Lx[:, IND].copy(), Ux[:, IND].copy()
        
    Lx[ind_l], Ux[ind_u] = tmp_l, tmp_u
    
    if len(p._discreteVarsNumList):
        Lx[:, IND], Ux[:, IND] = L, U
    
    for i in p._discreteVarsNumList:
        v = p._freeVarsList[i]
        
        ind_l1 = where(ind_l[:, i])[0]
        if ind_l1.size:
            mid1, mid2 = splitDomainForDiscreteVariable(Lx[ind_l1, i], Ux[ind_l1, i], v)
#                    Lx[ind_l1, i] = L[ind_l1, n+i]
            Lx[ind_l1, i] = mid2
#                    Lx[ind_l1, n+i] = mid2
            
        ind_u1 = where(ind_u[:, i])[0]
        if ind_u1.size:
            mid1, mid2 = splitDomainForDiscreteVariable(Lx[ind_u1, i], Ux[ind_u1, i], v)
            Ux[ind_u1, i] = mid1
#                    Ux[ind_u1, n+i] = U[ind_u1, i]
#                    Ux[ind_u1, i] = mid1

    # TODO: mb lock is required for parallel computations
    
    nlh_l, nlh_u = nlh[:, nlh.shape[1]//2:], nlh[:, :nlh.shape[1]//2]
    
    # inplace operations are performed in the cycle
    if 1:
        nlh_l[ind_u], nlh_u[ind_l] = nlh_u[ind_u].copy(), nlh_l[ind_l].copy()   
    else:
        ind_Tmp = logical_and(ind_u, logical_not(ind_l))
        nlh_l[ind_Tmp] = nlh_u[ind_Tmp].copy()
        ind_Tmp = logical_and(ind_l, logical_not(ind_u))
        nlh_u[ind_Tmp] = nlh_l[ind_Tmp].copy()
        
    return Lx, Ux, nlh, indT    




def getTruncatedArrays2(ind, Lx, Ux, indT, semiinvariant, nlh, nlh_0, ind_l = None, ind_u = None):
    # TODO: rework it when numpy will have appropriate inplace function
    Lx, Ux, indT, semiinvariant = getTruncatedArrays(ind, Lx, Ux, indT, semiinvariant)
    s = ind.size
    nlh_0 = nlh_0[ind]
    nlh = take(nlh, ind, axis=0, out=nlh[:s])
    
    if ind_l is None:
        return Lx, Ux, indT, semiinvariant, nlh, nlh_0

    ind_l = take(ind_l, ind, axis=0, out=ind_l[:s])
    ind_u = take(ind_u, ind, axis=0, out=ind_u[:s])
    return Lx, Ux, indT, semiinvariant, nlh, nlh_0, ind_l, ind_u


normalizer = (np.pi/2-np.arctan(range(5))).mean()

def getMultiplier(activeCons, c_id, iter):
#    return 1.0
    multiplier = np.empty(len(activeCons))
    for j, ac in enumerate(activeCons):
        trig_info_lst = ac.triggering_info.get(c_id, None)
        if trig_info_lst is None:
#            print('iter:', iter)
            multiplier[j] = 1.0
        else:
            multiplier[j] = (np.pi/2-np.arctan(iter - np.array(trig_info_lst))).mean() / normalizer
#            print(iter, np.array(trig_info_lst))
    return multiplier
    
    
    
