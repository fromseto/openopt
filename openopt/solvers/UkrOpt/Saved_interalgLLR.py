PythonSum = sum
PythonMax = max
from numpy import tile, isnan, array, atleast_1d, asarray, logical_and, all, logical_or, any, nan, isinf, \
arange, vstack, inf, logical_not, take, abs, hstack, empty, \
prod, int16, int32, int64, log2, searchsorted, cumprod#where

# for PyPy
from openopt.kernel.nonOptMisc import where, isPyPy
from bisect import bisect, bisect_left, bisect_right

import numpy as np
from FuncDesigner import oopoint
from FuncDesigner.multiarray import multiarray
from interalgT import *

if isPyPy:
    nanargmin = nanargmin_axis
    nanargmax = nanargmax_axis
else:
    try:
        from bottleneck import nanargmin
    except ImportError:
        from numpy import nanargmin

try:
    from bottleneck import nanmin, nanmax
except ImportError:
    from numpy import nanmin, nanmax
    
    
def nanargmin_axis(a, axis = None):
    if axis is None:
        ind = -1
        Min = inf
        for i, elem in enumerate(a.flat):
            if elem <= Min:
                ind = i
                Min = elem
        return np.nan if ind == -1 else ind
    
    assert axis in (0, 1), 'unimplemented yet'
    
    A = a if type(a) == np.ndarray and a.ndim > 1 else atleast_2d(a)
    
    inds = np.empty(A.shape[1-axis], int)
    inds.fill(-1)
    Mins = np.empty(A.shape[1-axis], A.dtype)
    if str(A.dtype).startswith('float'):
        Mins.fill(np.inf)
    elif axis == 0:
        Mins = A[:, 0].copy()
    else:
        Mins = A[0].copy()
    for i in range(A.shape[axis]):
        tmp = A[i] if axis == 0 else A[:, i]
        ind = tmp <= Mins
        Mins[ind] = tmp[ind]
        inds[ind] = i
    return inds

def nanargmax_axis(A, axis = None):
    if axis is None:
        ind = -1
        Max = -inf
        for i, elem in enumerate(A.flat):
            if elem >= Max:
                ind = i
                Max = elem
        return np.nan if ind == -1 else ind
    
    assert axis in (0, 1), 'unimplemented yet'
    
    inds = np.empty(A.shape[1-axis], int)
    inds.fill(-1)
    Maxs = np.empty(A.shape[1-axis], A.dtype)
    Maxs.fill(-np.inf)
    for i in range(A.shape[axis]):
        tmp = A[i] if axis == 0 else A[:, i]
        ind = tmp >= Maxs
        Maxs[ind] = tmp[ind]
        inds[ind] = i
    return inds
    
    
def getIntervals2(Lx, Ux, ooVars, f, dataType, p, Th = None, semiinvariant = None, indT = None):
    domain = oopoint(((v, (Lx[:, i], Ux[:, i])) for i, v in enumerate(ooVars)), skipArrayCast=True, isMultiPoint=True)
    domain.dictOfFixedFuncs = p.dictOfFixedFuncs
    domain._dictOfRedirectedFuncs = p._dictOfRedirectedFuncs
    domain._dictOfStochVars = p._dictOfStochVars
    domain.maxDistributionSize = p.maxDistributionSize
    domain.nPoints = Lx.shape[0]
    domain._p = p
    
    new = 0
    if new:
        r, r0, Lx, Ux, semiinvariant, indT = f.iqg(domain, dataType, UB = Th, Lx = Lx, Ux = Ux, 
                              semiinvariant = semiinvariant, indT = indT)
    else:
#        r, r0, _Lx, _Ux  = f.iqg(domain, dataType, UB = Th)
        r, r0  = f.iqg(domain, dataType, UB = Th)
    if r is None:
        return None, None, None, None, False, False, array([]), array([])
    
    Lf_l, Lf_u, Uf_l, Uf_u = [], [], [], []
    definiteRange = r0.definiteRange
    for v in ooVars:
        # TODO: rework and optimize it
        tmp = r.get(v, None)
        if tmp is not None:
            Lf_l.append(tmp[0].lb)
            Lf_u.append(tmp[1].lb)
            Uf_l.append(tmp[0].ub)
            Uf_u.append(tmp[1].ub)
            
            #TODO: mb don't compute definiteRange for modificationVar != None
#            definiteRange = logical_and(definiteRange, tmp[0].definiteRange)
#            definiteRange = logical_and(definiteRange, tmp[1].definiteRange)
        else:
            Lf_l.append(r0.lb)
            Lf_u.append(r0.lb)
            Uf_l.append(r0.ub)
            Uf_u.append(r0.ub)
            #definiteRange = logical_and(definiteRange, r0.definiteRange)
    Lf, Uf = hstack(Lf_l+Lf_u), hstack(Uf_l+Uf_u)    
    return Lx, Ux, Lf, Uf, definiteRange, domain.exactRange, semiinvariant, indT

def formIntervalPoint(Lx, Ux, ooVars):
    m, n = Lx.shape
    LB = [[] for i in range(n)]
    UB = [[] for i in range(n)]

    Centers = (Lx + Ux) / 2
    
    for i in range(n):
        t1, t2 = tile(Lx[:, i], 2*n), tile(Ux[:, i], 2*n)
        t1[(n+i)*m:(n+i+1)*m] = t2[i*m:(i+1)*m] = Centers[:, i]
        
#        if ooVars[i].domain is bool:
#            indINQ = Lx[:, i] != Ux[:, i]
#            tmp = t1[(n+i)*m:(n+i+1)*m]
#            tmp[indINQ] = 1
#            tmp = t2[i*m:(i+1)*m]
#            tmp[indINQ] = 0
            
#        if ooVars[i].domain is bool:
#            t1[(n+i)*m:(n+i+1)*m] = 1
#            t2[i*m:(i+1)*m] = 0
#        else:
#            t1[(n+i)*m:(n+i+1)*m] = t2[i*m:(i+1)*m] = Centers[:, i]
        
        LB[i], UB[i] = t1, t2
    
    domain = dict((v, (LB[i], UB[i])) for i, v in enumerate(ooVars))
    domain = oopoint(domain, skipArrayCast = True)
    domain.isMultiPoint = True
    domain.nPoints = 2*n*m#LB[0].shape[0]#2*n#4 * m#2*n#
    return domain

def getIntervals(domain, func, dataType):
    TMP = func.interval(domain, dataType)
    #assert TMP.lb.dtype == dataType
    return asarray(TMP.lb, dtype=dataType), asarray(TMP.ub, dtype=dataType), TMP.definiteRange

def getCentersValues(ooVars, Lx, Ux, activeCons, tnlh, func, C, contol, dataType, p, threshold = inf):
    #TODO: use threshold with objective before constraints when objective is simpler than constraints
    m, n = Lx.shape
    m0 = m
    wCenters_0 = (Lx+Ux) / 2
    if tnlh is None:# or p.probType =='MOP':
        wCenters = wCenters_0
    else:
        tnlh = tnlh.copy()
        tnlh[atleast_1d(tnlh<1e-300)] = 1e-300 # to prevent division by zero
        tnlh[atleast_1d(isnan(tnlh))] = inf #- check it!
        tnlh_l_inv, tnlh_u_inv = 1.0 / tnlh[:, :n], 1.0 / tnlh[:, n:]
        wCenters = (Lx * tnlh_l_inv + Ux * tnlh_u_inv) / (tnlh_l_inv + tnlh_u_inv)
        ind = tnlh_l_inv == tnlh_u_inv # especially important for tnlh_l_inv == tnlh_u_inv = 0
        wCenters[ind] = (Lx[ind] + Ux[ind]) / 2
        wCenters_1 = (wCenters + wCenters_0)/2
        wCenters = vstack((wCenters, wCenters_0, wCenters_1))
    # TODO: calculate centers in new way (median instead of rounding (lx + Ux) / 2)
    adjustCentersWithDiscreteVariables(wCenters, p)
    if p._isStochastic:
        centers = dict((oovar, asarray(wCenters[:, i], dataType).reshape(-1, 1).view(multiarray)) for i, oovar in enumerate(ooVars))
    else:
        centers = dict((oovar, asarray(wCenters[:, i], dataType)) for i, oovar in enumerate(ooVars))
        
    centers = oopoint(centers, skipArrayCast = True, maxDistributionSize = p.maxDistributionSize)
    centers.isMultiPoint = True
    centers.dictOfFixedFuncs = p.dictOfFixedFuncs
    centers._dictOfRedirectedFuncs = p._dictOfRedirectedFuncs
    centers._dictOfStochVars = p._dictOfStochVars
    centers.nPoints = m
    
    m = wCenters.shape[0]
    
    kw =  {'Vars': p.freeVars} if p.freeVars is None or (p.fixedVars is not None and len(p.freeVars) < len(p.fixedVars)) else {'fixedVars': p.fixedVars}
    kw['fixedVarsScheduleID'] = p._FDVarsID

    if len(C) != 0:
        indConstraints = empty(m, bool)
        indConstraints.fill(True)
        for _c, f, lb_, ub_ in C.values():
            c = f(centers, **kw).flatten()
#            print(c,lb_,ub_)
            ind_feasible = logical_and(c  >= lb_, c <= ub_) # here lb_ and ub_ are already shifted by required tolerance
            indConstraints = logical_and(indConstraints, ind_feasible)
            
            # changes
            # TODO: rework the cycle to speedup it
            if p.solver.prioritized_constraints:
                for k in range(m//m0):
                    for j in where(logical_not(ind_feasible[k*m0:(k+1)*m0]))[0]:
                        triggering_dict = activeCons[j].triggering_info
                        id = _c._id
                        tmp = triggering_dict.get(id, [])
                        if len(tmp) == 0:
                            triggering_dict[id] = [p.iter]
                        else:
    #                        tmp2 = triggering_dict[id]
                            if len(tmp) > 5*k:
                                tmp.pop(0)
                            tmp.append(p.iter)
#            tmp = activeCons.
            # changes end
            
            
    else:
        indConstraints = True
#    print('constr ind:',indConstraints)
        
    isMOP = p.probType =='MOP'
    if not any(indConstraints):
        F = empty(m, dataType)
        #TODO: replace 2**n by nan when integer nan in Python will be available
        F.fill(2**31-2 if dataType in (int16, int32, int64, int) else nan) # mb add int8
        if isMOP:
            FF = [F for k in range(p.nf)]
    elif all(indConstraints):
        if isMOP:
            FF = [fun(centers, **kw) for fun in func]
        else:
            F = atleast_1d(asarray(func(centers, **kw), dataType).view(np.ndarray)).flatten() \
            if func is not None else np.zeros(m) # func can be None for SNLE, TODO: check is it still remain
    else:
        #centers = dict([(oovar, (Lx[indConstraints, i] + Ux[indConstraints, i])/2) for i, oovar in enumerate(ooVars)])
        #centers = ooPoint(centers, skipArrayCast = True)
        #centers.isMultiPoint = True
        if isMOP:
            FF = []
            for fun in func:
                tmp = atleast_1d(asarray(fun(centers, **kw), dataType).view(np.ndarray)).flatten() 
                F = empty(m, dataType)
                F.fill(2**31-15 if dataType in (int16, int32, int64, int) else nan)
                F[indConstraints] = tmp[indConstraints]    
                FF.append(F)
        else:
            tmp = atleast_1d(asarray(func(centers, **kw), dataType).view(np.ndarray)).flatten() if func is not None \
            else np.zeros(m) # func can be None for SNLE, TODO: check is it still remain
            
            F = empty(m, dataType)
            F.fill(2**31-15 if dataType in (int16, int32, int64, int) else nan)
            F[indConstraints] = tmp[indConstraints]
    #TODO: reduce centers by ind before computations
    if isMOP:
        return array(FF).view(np.ndarray).T.reshape(m, len(func)).tolist(), wCenters.tolist()
    else:
        return F, wCenters


def adjustCentersWithDiscreteVariables(wCenters, p):
    # TODO: rework it for discrete variables
    for i in p._discreteVarsNumList:
        v = p._freeVarsList[i]
        
        if v.domain is bool or v.domain is 'bool':
            wCenters[:, i] = where(wCenters[:, i]<0.5, 0, 1)
        else:
            tmp = wCenters[:, i]
            d = v.domain 
            if isPyPy:
                d2 = d.tolist()
                ind = atleast_1d([bisect_left(d2, val) for val in tmp])
                ind2 = atleast_1d([bisect_right(d2, val) for val in tmp])
            else:
                ind = searchsorted(d, tmp, side='left')
                ind2 = searchsorted(d, tmp, side='right')
            ind3 = where(ind!=ind2)[0]
            Tmp = tmp[ind3].copy()
            
            ind[ind==d.size] -= 1 # may be due to roundoff errors
            ind[ind==1] = 0
            ind2[ind2==d.size] -=1
            ind2[ind2==0] = 1 # may be due to roundoff errors
            tmp1 = asarray(d[ind], p.solver.dataType)
            tmp2 = asarray(d[ind2], p.solver.dataType)
            if Tmp.size!=0:
                if str(tmp1.dtype).startswith('int'):
                    Tmp = asarray(Tmp, p.solver.dataType)
                tmp2[ind3] = Tmp.copy()
                tmp1[ind3] = Tmp.copy()
            tmp = where(abs(tmp-tmp1)<abs(tmp-tmp2), tmp1, tmp2)
            #print max(abs(tmp-tmp1)), max(abs(tmp-tmp2))
            wCenters[:, i] = tmp
    #print where(wCenters==0)[0].size, where(wCenters==1)[0].size

def getBestCenterAndObjective(PointVals, PointCoords, dataType):
#    bestCenterInd = nanargmin(PointVals)
#    if isnan(bestCenterInd):
    # for bottleneck v >= 0.8:
    bestCenterInd = 0 if np.all(isnan(PointVals)) else nanargmin(PointVals)
    # TODO: check it , maybe it can be improved
    #bestCenter = centers[bestCenterInd]
    #bestCenterCoords = array([(val[0][bestCenterInd]+val[1][bestCenterInd]) / 2 for val in domain.values()], dtype=dataType)
    #bestCenterObjective = atleast_1d(FuncVals)[bestCenterInd] if not isnan(bestCenterInd) else inf
    bestCenterCoords = array(PointCoords[bestCenterInd], dtype=dataType)
    bestCenterObjective = atleast_1d(PointVals)[bestCenterInd] 
    return bestCenterCoords, bestCenterObjective
    
def getSomeNodes(AllNodes, maxActiveNodes, p):

#    dataHandling = p.solver.dataHandling
    minActiveNodes = p.solver.minActiveNodes
    m = len(AllNodes)
    if m <= maxActiveNodes:
        return AllNodes, []#array([], object)
    
    activeNodes, inactiveNodes = AllNodes[:maxActiveNodes], AllNodes[maxActiveNodes:]    
        
    if getattr(activeNodes[0], 'tnlh_curr_best', None) is not None:
        #t0 = activeNodes[0].tnlh_curr_best
        tnlh_curr_best_values = asarray([node.tnlh_curr_best for node in activeNodes])
        
        #changes
        tmp = 2 ** (-tnlh_curr_best_values)
        Tmp = -cumprod(1.0-tmp)
        
        if isPyPy:
            ind2 = bisect(Tmp.tolist(), -0.05)
        else:
            ind2 = searchsorted(Tmp, -0.05)
        #changes end

        ind = ind2
        
        M = max((minActiveNodes, ind))
        
#        # IMPORTANT!
#        if M == 0: M = 1
        
        tmp1, tmp2 = activeNodes[:M], activeNodes[M:]
        activeNodes = tmp1
        inactiveNodes = tmp2 + inactiveNodes#hstack((tmp2, inactiveNodes))
        
    # TODO: implement it for MOP as well
#    cond_min_uf = 0 and dataHandling == 'raw' and hasattr(AllNodes[0], 'key')            
#    
#    if cond_min_uf:
#        num_nlh = min((max((1, int(0.8*maxActiveNodes))), len(activeNodes)))
#        num_uf = min((maxActiveNodes - num_nlh, int(maxActiveNodes/2)))
#        if num_uf < 15:
#            num_uf = 15
#        #activeNodes, inactiveNodes = AllNodes[:num_nlh], AllNodes[num_nlh:]    
#        Ind = np.argsort([node.key for node in inactiveNodes])
#        
#        min_uf_nodes = inactiveNodes[Ind[:num_uf]]
#        inactiveNodes = inactiveNodes[Ind[num_uf:]]
#        
#        activeNodes = activeNodes + min_uf_node#np.hstack((activeNodes, min_uf_nodes))
        
    # changes end
    #print maxActiveNodes, len(activeNodes), len(inactiveNodes)
    
    return activeNodes, inactiveNodes

def getBestCoordsForSplitting(
        tnlhf_curr, nlhc, nlh_obj_fixed, residual, 
        Lx, Ux, Lf, Uf, semiinvariant_prev, p, indT, activeCons):
    
    m, n = Lx.shape
    arangeM = arange(m)
    
    if p.probType == 'IP':
        Lfc_modL, Lfc_modU = Lf[:, :n], Lf[:, n:]
        Ufc_modL, Ufc_modU = Uf[:, :n], Uf[:, n:]
#            # TODO: handle nans
        minLf = where(Lfc_modL < Lfc_modU, Lfc_modL, Lfc_modU)
        maxUf = where(Ufc_modL < Ufc_modU, Ufc_modU, Ufc_modL)
    
        # Prev
        tmp = Uf[:, 0:n]-Lf[:, 0:n]+Uf[:, n:]-Lf[:, n:]
        bestCoordsForSplitting = nanargmin(tmp,1)
        d = 0.5*tmp[arangeM, bestCoordsForSplitting]
        
        
        #New
#        tmp = Uf - Lf
#        bestCoordsForSplitting_ = nanargmin(tmp,1)
#        bestCoordsForSplitting = bestCoordsForSplitting_% n
#        d = tmp[arangeM, bestCoordsForSplitting_]

#        ind = 2**(-n) >= (semiinvariant_prev - d)/asarray(d, 'float64')
        ind_problematic = 2**(1.0/n) * d >= semiinvariant_prev
        #new
#        ind = 2**(1.0/n) * d >= nanmax(maxUf-minLf, 1)
        
        #ind = 2**(-n) >= (semiinvariant_prev - semiinvariant)/asarray(semiinvariant, 'float64')
    
        #s2 = nanmin(maxUf - minLf, 1)
        #print (abs(s2/semiinvariant))
        
        # Prev
        semiinvariant = nanmin(maxUf - minLf, 1)
        
        # New
        #semiinvariant = nanmax(maxUf - minLf, 1)
#        semiinvariant = nanmax(Uf - Lf, 1)
        
        #ind = semiinvariant_prev  <= semiinvariant + ((2**-n / log(2)) if n > 15 else log2(1+2**-n)) 
        indD = None
        #print len(where(indD)[0]), len(where(logical_not(indD))[0])
#    elif p.probType == 'MOP':
#
#        raise 'unimplemented'
    else:
        if p.solver.dataHandling == 'sorted':
            semiinvariant = getSemiinvariant(Lf, Uf)
            
            bestCoordsForSplitting = nanargmin(Uf, 1) % n

            d = nanmax([Uf[arangeM, bestCoordsForSplitting] - Lf[arangeM, bestCoordsForSplitting], 
                    Uf[arangeM, n+bestCoordsForSplitting] - Lf[arangeM, n+bestCoordsForSplitting]], 0)
            
            ## !!!! Don't replace it by (semiinvariant_prev /d- 1) to omit rounding errors ###
            #ind = 2**(-n) >= (semiinvariant_prev - d)/asarray(d, 'float64')
            
            #NEW
            ind_problematic = d  >=  semiinvariant_prev / 2 ** (1.0e-1 / n)
            #ind = d  >=  semiinvariant_prev / 2 ** (1.0/n)
            indD = empty(m, bool)
            indD.fill(True)
            #ind.fill(False)
            ###################################################
        elif p.solver.dataHandling == 'raw':
            
            # !!!!!!!!
            # TODO: mb use personal len(dep) (by obj, cons dependencies) instead of n here
            # (however, it's costly)
            # !!!!!!!!
            
            if 1:
                delta = 1.0/n #PythonMax(n, 2)
            else:
#                delta = 
                pass
            
#            case = 'new'
#            if case == 'prev' and p.probType == 'MOP':
#                bestCoordsForSplitting = p._bestCoordsForSplitting[:m]
#                p._bestCoordsForSplitting = p._bestCoordsForSplitting[m:]
#                d = semiinvariant = p._semiinvariant[:m]
#                p._semiinvariant = p._semiinvariant[m:]
#                ind_problematic = semiinvariant_prev  <= d + delta
#            else:
            
            if p.probType == 'MOP':
                bestCoordsForSplitting = p._bestCoordsForSplitting[:m]
                p._bestCoordsForSplitting = p._bestCoordsForSplitting[m:]                    
            else:
                T = tnlhf_curr
                tnlh_curr_1, tnlh_curr_2 = T[:, :n], T[:, n:]
                TNHL_curr_min = where(logical_or(tnlh_curr_1 < tnlh_curr_2, isnan(tnlh_curr_2)), tnlh_curr_1, tnlh_curr_2)
                bestCoordsForSplitting = nanargmin(TNHL_curr_min, 1)
            
            # 1. Processing objective
            T = nlh_obj_fixed
            s_obj = nanmin(vstack((T[arangeM, bestCoordsForSplitting], T[arangeM, n+bestCoordsForSplitting])), 0)
            
#            if p.probType != 'ODE':
#                # TODO: mb use min/max instead of nanmin/nanmax
#                maxUf, minLf = nanmax(Uf, axis=1), nanmin(Lf, axis=1) #TODO: mb improve it (computate it wrt modL/modU)
#                ind_insufficient_obj_diff = maxUf - minLf < p.fTol
            
            # 2. Processing constraints
            #TODO: SKIP nlhc IF ALL ELEMENTS ARE _constraintInactiveValue (OR mb if activeCons are empty lists)
            T = nlhc
            s_c = nanmin(vstack((T[arangeM, bestCoordsForSplitting], T[arangeM, n+bestCoordsForSplitting])), 0)
            
            # 3. Gathering results
            s_obj_prev, s_c_prev = np.real(semiinvariant_prev), np.imag(semiinvariant_prev)
            ind_problematic_c = logical_and(s_c_prev  <= s_c + delta, s_c != p.solver._constraintInactiveValue)
            
            ind_problematic_obj = s_obj_prev  <= s_obj + delta
            ind_problematic = logical_and(ind_problematic_obj, ind_problematic_c)
            semiinvariant = np.atleast_1d(s_obj+ 1j * s_c)
            
            #OLD
            #!#!#!#! Don't replace it by semiinvariant_prev - d <= ... to omit inf-inf = nan !#!#!#
            #ind = semiinvariant_prev  <= d + ((2**-n / log(2)) if n > 15 else log2(1+2**-n)) 
            #ind = semiinvariant_prev - d <= ((2**-n / log(2)) if n > 15 else log2(1+2**-n)) 
            
            #NEW
            indD = logical_or(ind_problematic, logical_not(indT))
#            indQ = d >= semiinvariant_prev - delta
#            indD = logical_or(indQ, logical_not(indT))
        else:
            assert 0

    if any(ind_problematic):
        IND_problematic = where(ind_problematic)[0]
        boxShapes = Ux[IND_problematic] - Lx[IND_problematic]
        bestCoordsForSplitting[IND_problematic] = np.argmax(boxShapes, 1) 
        
        # TODO: mb handle IP and ODE here
        skipProbs = ('IP', 'ODE')
        # TODO: mb handle case p.solver.dataHandling == 'sorted' here as well
        if 1 and p.probType not in skipProbs and p.solver.dataHandling == 'raw':
            ind_ignore_obj = logical_not(ind_problematic_obj[IND_problematic])
            
            if p.probType != 'MOP':
                UfLfDiff = Uf[IND_problematic] - Lf[IND_problematic]
                
                # TODO: mb use the code below instead of mere nanmax(UfLfDiff)?
#            diff_mod_l, diff_mod_u = UfLfDiff[:, :n], UfLfDiff[:, n:]
#            max_UfLf_diff = where(diff_mod_l > diff_mod_u, diff_mod_l, diff_mod_u) # TODO: mb check for NaNs here
#            Max_UfLf_diff = np.max(max_UfLf_diff, axis=1)
                Max_UfLf_diff = np.nanmax(UfLfDiff, axis=1)#TODO: mb use max instead of nanmax (faster but less stable wrt possible future changes)
                
                ind_small_delta = Max_UfLf_diff < p.fTol # TODO: mb handle rTol here as well
                ind_ignore_obj |= ind_small_delta
                #  TODO: mb rework it, now it's just ind_ignore_obj = ind_small_delta
                
                
            #TODO: mb it can be reduced via skipping ind_small_delta processing

            #can be vectorized, but difference is insufficient
            constraints_dep_dict = p._constraints_dep_dict
            freeVarsDict = p._freeVarsDict
            objdep = p._objDep
            
            for j, i in enumerate(IND_problematic):
                if ind_ignore_obj[j]:
                    dep = set()
                elif len(objdep) != n:
                    dep = objdep.copy()
                else:
                    continue
                C = [constraints_dep_dict[c_id] for c_id in activeCons[i]]
                dep.update(*C)
                
                # TODO: is it required? well let's have it for more safety wrt future changes
                if len(dep) == 0: 
                    p.debugmsg('!!! Warning! len(dep) is zero in interalgLLR.getBestCoordsForSplitting()')
                    continue 
                
                # TODO: speedup inds_active obtaining
                inds_active = array([freeVarsDict[v] for v in dep])
                _ind = np.argmax(boxShapes[j, inds_active])
                _i = inds_active[_ind]
                bestCoordsForSplitting[i] = _i
    
    return bestCoordsForSplitting, semiinvariant, indD
    
def getSemiinvariant(Lf, Uf): 
    m, n = Lf.shape
    n //= 2
#    if case == 1:
#        U1, U2 = Uf[:, :n].copy(), Uf[:, n:] 
#        #TODO: mb use nanmax(concatenate((U1,U2),3),3) instead?
#        U1 = where(logical_or(U1<U2, isnan(U1)),  U2, U1)
#        return nanmin(U1, 1)
        
    L1, L2, U1, U2 = Lf[:, :n], Lf[:, n:], Uf[:, :n], Uf[:, n:] 
#    if case == 2:
    U = where(logical_or(U1<U2, isnan(U1)),  U2, U1)
    L = where(logical_or(L2<L1, isnan(L1)), L2, L1)
    return nanmax(U-L, axis=1)

def formNewBoxes(Lx, Ux, bestCoordsForSplitting, ooVars):#, tnlh_fixed_curr):
    #assert Lx.size != 0
    if Lx.size == 0: return Lx.copy(), Ux.copy() # especially for PyPy
    new_Lx, new_Ux = Lx.copy(), Ux.copy()
    m, n = Lx.shape
    arangeM = arange(m)
    
    #!!!! TODO: rework it for discrete problems    
    newCoords = (new_Lx[arangeM, bestCoordsForSplitting] + new_Ux[arangeM, bestCoordsForSplitting]) / 2
    
    ### !!!!!!!!!!!!!!!!!!!!!
    # TODO: rework it, don't recalculate each time
    DiscreteVars =  [v.domain is not None for v in ooVars] 
    BoolVars = [(v.domain is bool or v.domain is 'bool') for v in ooVars]

    indDiscrete = where(DiscreteVars)[0]
    discreteCoordsSet = set(indDiscrete)
    
    if 1 and all(BoolVars):
        # TODO: use it where Lx != Ux
        ind = Lx[arangeM, bestCoordsForSplitting] != Ux[arangeM, bestCoordsForSplitting]
        if any(ind):
            arangeM2, bestCoordsForSplitting2 = arangeM[ind], bestCoordsForSplitting[ind]
            new_Lx[arangeM2, bestCoordsForSplitting2] = 1
            new_Ux[arangeM2, bestCoordsForSplitting2] = 0
    
    else:
        new_Lx[arangeM, bestCoordsForSplitting] = newCoords
        new_Ux[arangeM, bestCoordsForSplitting] = newCoords
        
        # TODO: mb rework it
        for ind, coord in enumerate(bestCoordsForSplitting):
            if coord in discreteCoordsSet:
                #d = ooVars[coord]
                # = (0, 1) if d is bool or d is 'bool' else\
                m1, m2 = \
                splitDomainForDiscreteVariable(array([Lx[ind, coord]]), array([Ux[ind, coord]]), ooVars[coord])
                
                new_Lx[ind, coord] = m2
                new_Ux[ind, coord] = m1

    new_Lx = vstack((Lx, new_Lx))
    new_Ux = vstack((new_Ux, Ux))

    
#    if tnlhf_curr is not None:
#        tnlhf_curr_local = hstack((tnlhf_curr[arangeM, bestCoordsForSplitting], tnlhf_curr[arangeM, n+bestCoordsForSplitting]))
#    else:
#        tnlhf_curr_local = None
    return new_Lx, new_Ux#, tnlhf_curr_local


def splitNodes(AllNodes, maxActiveNodes, p, Solutions, ooVars, varTols, threshold):
    solutions, SolutionCoords = Solutions.solutions, Solutions.coords

    if len(AllNodes) == 0:
        return array([]), array([]), [], array([]), []
    inactiveNodes = AllNodes
    
    if SolutionCoords.size != 0:
        solutionCoordsL, solutionCoordsU = SolutionCoords - varTols, SolutionCoords + varTols
    Lx, Ux, S, AC = [], [], [], []
    
    #Tnlhf_curr_local = []
#    n = p.n
    N = 0
    maxSolutions = p.maxSolutions
#    constraints_counter = p._constraints_counter
    
#    new = 1
#    # new
#    if new and p.probType in ('MOP', 'SNLE', 'GLP', 'NLP', 'MINLP') and p.maxSolutions == 1:
#        
#        
#        return Lx, Ux, inactiveNodes, semiinvariant
        
    
    while True:
        activeNodesCandidates, inactiveNodes = getSomeNodes(inactiveNodes, maxActiveNodes, p)

        #print nanmax(2**(-activeNodesCandidates[0].tnlh_curr)) ,  nanmax(2**(-activeNodesCandidates[-1].tnlh_curr))
        Lxc, Uxc, Lfc, Ufc, SIc = asarray([t.Lx for t in activeNodesCandidates]), \
        asarray([t.Ux for t in activeNodesCandidates]), \
        asarray([t.Lf for t in activeNodesCandidates]), \
        asarray([t.Uf for t in activeNodesCandidates]), \
        asarray([t.semiinvariant for t in activeNodesCandidates])
        
        activeCons = [t.activeCons for t in activeNodesCandidates]
        
        if p.probType == 'MOP':
            
            # 
            # !!!!!!!! TODO: CHECK  tnlhf_curr <- tnlh_all 
            # VS tnlhf_curr <- tnlh_curr in next section
            #
            if p.solver.mop_mode == 1:
                tnlhf_curr = asarray([t.tnlh_all for t in activeNodesCandidates])
            else:
                tnlhf_curr = None
            nlhc = asarray([t.nlhc for t in activeNodesCandidates]) 
#            nlh_obj_fixed = None
            nlh_obj_fixed = asarray([t.nlh_obj_fixed for t in activeNodesCandidates]) 
        elif p.solver.dataHandling == 'raw':
            tnlhf_curr = asarray([t.tnlh_curr for t in activeNodesCandidates]) 
            nlhc = asarray([t.nlhc for t in activeNodesCandidates]) 
            nlh_obj_fixed = asarray([t.nlh_obj_fixed for t in activeNodesCandidates]) 
        else:
            tnlhf_curr, nlhc, nlh_obj_fixed = None, None, None
        
        
        if p.probType != 'IP': 
            #nlhc = asarray([t.nlhc for t in activeNodesCandidates])
            
            #residual = asarray([t.residual for t in activeNodesCandidates]) 
            residual = None
            
            # TODO: implement it in interalgMOP code
#            if p.probType != 'MOP':
            if p.solver.mop_mode == 1:
                indT = TruncateSomeBoxes(p, Lxc, Uxc, Lfc, Ufc, threshold, tnlhf_curr)
                if activeNodesCandidates[0].indtc is not None:
                    indtc = asarray([t.indtc for t in activeNodesCandidates])
                    indT = logical_or(indT, indtc)
            else:
                indT = asarray([t.indtc for t in activeNodesCandidates])
        else:
            residual = None
            indT = None
        bestCoordsForSplitting, semiinvariant, indD = getBestCoordsForSplitting(tnlhf_curr, nlhc, nlh_obj_fixed, residual, Lxc, Uxc, Lfc, Ufc, SIc, p, indT, activeCons)
        
        NewD = 1
        
        if NewD and indD is not None:#TODO: mb handle case  all(indD) here and skip to part 2
            ind_divide, ind_remain = where(indD)[0], where(logical_not(indD))[0]
            s4d = semiinvariant[ind_divide]
            sf = semiinvariant[ind_remain]
            AC_divide, AC_remain = [activeCons[i] for i in ind_divide],\
            [activeCons[i] for i in ind_remain]

            semiinvariant = hstack((s4d, s4d, sf))
            
            #activeCons = 2*AC_divide + AC_remain
            activeCons = AC_divide+[elem.copy() for elem in AC_divide] + AC_remain
            
            Lxf, Uxf = Lxc[ind_remain], Uxc[ind_remain]
            Lxc, Uxc = Lxc[ind_divide], Uxc[ind_divide]
            bestCoordsForSplitting = bestCoordsForSplitting[ind_divide]
            
        else:
            semiinvariant = tile(semiinvariant, 2)
            #activeCons = 2*activeCons # Python list duplication
            activeCons = activeCons + [elem.copy() for elem in activeCons]

        #Lxc, Uxc, tnlhf_curr_local = formNewBoxes(Lxc, Uxc, bestCoordsForSplitting, ooVars, tnlhf_curr)
        Lxc, Uxc = formNewBoxes(Lxc, Uxc, bestCoordsForSplitting, ooVars)

        if NewD and indD is not None:
            Lxc = vstack((Lxc, Lxf))
            Uxc = vstack((Uxc, Uxf))
            
        if maxSolutions == 1 or len(solutions) == 0: 
            #Lx, Ux, Tnlhf_curr_local = Lxc, Uxc, tnlhf_curr_local
            Lx, Ux = Lxc, Uxc
            break
        
        # TODO: change cycle variable if len(solutions) >> maxActiveNodes
        for i in range(len(solutions)):
            ind = logical_and(all(Lxc >= solutionCoordsL[i], 1), all(Uxc <= solutionCoordsU[i], 1))
            if any(ind):
                j = where(logical_not(ind))[0]
                lj = j.size
                Lxc = take(Lxc, j, axis=0, out=Lxc[:lj])
                Uxc = take(Uxc, j, axis=0, out=Uxc[:lj])
                semiinvariant = semiinvariant[j]
                
#                for i in where(ind)[0]:
#                    box_active_constraints = activeCons[i]
#                    for c_id in box_active_constraints:
#                        constraints_counter[c_id] -= 1 
                activeCons = [activeCons[idx] for idx in j]
                
#                if tnlhf_curr_local is not None:
#                    tnlhf_curr_local = tnlhf_curr_local[j]
        Lx.append(Lxc)
        Ux.append(Uxc)
        S.append(semiinvariant)
        AC.append(activeCons)
        #Tnlhf_curr_local.append(tnlhf_curr_local)
        N += Lxc.shape[0]
        if len(inactiveNodes) == 0 or N >= maxActiveNodes: 
            Lx, Ux, semiinvariant = vstack(Lx), vstack(Ux), hstack(S)
            activeCons = PythonSum(AC, []) 
            #Tnlhf_curr_local = hstack(Tnlhf_curr_local)
            break
    
    assert len(activeCons) == semiinvariant.size
    
    return Lx, Ux, inactiveNodes, semiinvariant, activeCons

Fields = ['key', 'Lx', 'Ux', 'nlh_obj_fixed','nlhc', 'indtc','residual', 'Lf', 'Uf', \
'semiinvariant', 'activeCons']
MOP_Fields = Fields[1:]
IP_fields = ['key', 'Lx', 'Ux', 'Lf', 'Uf', 'semiinvariant', 'F', 'volume', 'volumeResidual', 'activeCons']

# TODO: add activeCons when cons will be implemented for IP

def formNodes(Lx, Ux, nlhc, indT, residual, Lf, Uf, semiinvariant, p, activeCons, ind_nan = None): 
    Lx, Ux, Lf, Uf, semiinvariant, indT, nlhc, residual, activeCons = \
    remove_NaN_nodes(Lx, Ux, Lf, Uf, semiinvariant, indT, nlhc, residual, activeCons, p._constraints_counter, ind_nan)
    
    m, n = Lx.shape
    if m == 0:
        assert p.probType not in ('IP', 'ODE'), 'unimplemented for functions with domain including NaNs yet'
        return [], Lx, Ux, Lf, Uf, semiinvariant, indT, nlhc, residual, activeCons
        
    assert m == semiinvariant.size == len(activeCons)
    if p.probType == "IP":
        arangeM = arange(m)
        # TODO: omit recalculation from getBestCoordsForSplitting
        ind = nanargmin(Uf[:, 0:n] - Lf[:, 0:n] + Uf[:, n:] - Lf[:, n:], 1)
        sup_inf_diff = 0.5*(Uf[arangeM, ind] - Lf[arangeM, ind] + Uf[arangeM, n+ind] - Lf[arangeM, n+ind])
        volume = prod(Ux-Lx, 1)
        volumeResidual = volume * sup_inf_diff
        F = (Uf[arangeM, ind] + Lf[arangeM, ind] + Uf[arangeM, n+ind] + Lf[arangeM, n+ind]) / 4.0
        
        return [StoreItem(p, IP_fields, sup_inf_diff[i], Lx[i], Ux[i], Lf[i], Uf[i], semiinvariant[i], 
        F[i], volume[i], volumeResidual[i], activeCons[i]) for i in range(m)], Lx, Ux, Lf, Uf, semiinvariant, indT, nlhc, residual, activeCons
        

    residual = None
    
    isSNLE = p.probType in ('SNLE', 'NLSP')
    if 1 or not isSNLE:
        Lf, Uf = asarray(Lf), asarray(Uf)
        
        ind_inf = Uf==inf
        if ind_inf.any(): # without this check doesn't work with integer types
            Uf[ind_inf] = 1e300
        ind_inf = Lf==inf
        if ind_inf.any(): # without this check doesn't work with integer types
            Lf[Lf==-inf] = -1e300
        tmp = Uf - Lf
        
        
        #changes!!
        tmp[tmp<1e-300] = 1e-300
        #tmp[tmp<p.fTol] = p.fTol # TODO: mb rework it, e.g. replace by 1e-300
        
        
        
#            ind_uf_inf = where(Uf==inf)[0]
#            if ind_uf_inf.size:
#                Tmp = Lf[ind_uf_inf]
#                Tmp[Tmp==-inf] = -1e100
#                M = nanmax(abs(Tmp))
#                if M is nan or M == 0.0: 
#                    M = 1.0
#                tmp[ind_uf_inf] = 1e200 * (1.0 + Tmp/M)
        nlh_obj_fixed = log2(tmp)#-log2(p.fTol)
        
#        nlh_obj_fixed[Uf==inf] = 1e300# to make it not inf and nan
#        nlh_obj_fixed[Lf==-inf] = 1e300# to make it not inf and nan
    
    z = np.empty(2*n) # TODO: use None instead (mb replace by vector while encoding)
    z.fill(p.solver._constraintInactiveValue)# ~= log2(1e-300)
    
    if p.probType == "MOP":
        assert nlh_obj_fixed.ndim == 3, 'bug in interalg' 
        nlh_obj_fixed = nlh_obj_fixed.sum(axis=1)
        return [StoreItem(
                        p, MOP_Fields, Lx[i], Ux[i], nlh_obj_fixed[i], 
                        nlhc[i] if nlhc is not None else z, 
                        indT[i] if indT is not None else True, 
                        residual[i] if residual is not None else None, 
                        [Lf[i][k] for k in range(p.nf)], 
                        [Uf[i][k] for k in range(p.nf)], 
                        semiinvariant[i], 
                        activeCons[i] #if activeCons is not None else None
                        ) for i in range(m)], Lx, Ux, Lf, Uf, semiinvariant, indT, nlhc, residual, activeCons

    assert p.probType in ('GLP', 'NLP', 'NSP', 'SNLE', 'NLSP', 'MINLP', 'QP', 'LP', 'MILP')
    
    if 0 and isSNLE:
        nlh_obj_fixed = Tmp = Lf = Uf = [None]*m
    else:
        Lf_modL, Lf_modU = Lf[:, :n], Lf[:, n:]
        Tmp = nanmax(where(Lf_modU<Lf_modL, Lf_modU, Lf_modL), 1)
#        print('Lx:', Lx)
#        print('Ux:', Ux)
#        print('Lf:', Lf)
#        print('Tmp:', Tmp)
        # TODO: mb rework it
        nlh_obj_fixed[logical_and(isinf(Uf), isinf(nlh_obj_fixed))] = 1e300
    
#            residual = None

    return [StoreItem(p, Fields, Tmp[i], Lx[i], Ux[i], nlh_obj_fixed[i], 
                  nlhc[i] if nlhc is not None else z, 
                  indT[i] if indT is not None else None, 
                  residual[i] if residual is not None else None, 
                  Lf[i], Uf[i], semiinvariant[i], 
                  activeCons[i] #if activeCons is not None else None
                  ) for i in range(m)], Lx, Ux, Lf, Uf, semiinvariant, indT, nlhc, residual, activeCons


class StoreItem:
    def __init__(self, p, fields, *args, **kwargs):
#        self.p = p
        self._constraints_counter = p._constraints_counter
        self._constraints_reduction = p.solver._constraints_reduction
        for i in range(len(fields)):
            setattr(self, fields[i], args[i])
    
    def __del__(self):
        
        if self._constraints_reduction not in (1,  True):
            return
            
        tmp = getattr(self, 'activeCons', [])
        if len(tmp) == 0: 
            return
        
        constraints_counter = self._constraints_counter
#        print('1. constraints_counter:',constraints_counter)
        for item in tmp:
            constraints_counter[item] -= 1
#        print('2. constraints_counter:',constraints_counter)

#        new = 0
#        nn = 0
#if new and p.probType in ('MOP', 'SNLE', 'NLSP','GLP', 'NLP', 'MINLP') and p.maxSolutions == 1:
#            arr = tnlhf_curr if p.solver.dataHandling == 'raw' else Lfc
#            M = arr.shape[0]
#            arangeM = arange(M)
#            Midles = 0.5*(Lxc[arangeM, bestCoordsForSplitting] + Uxc[arangeM, bestCoordsForSplitting])
#            arr_1, arr2 = arr[arangeM, bestCoordsForSplitting], arr[arangeM, n+bestCoordsForSplitting]
#            Arr = hstack((arr_1, arr2))
#            ind = np.argsort(Arr)
#            Ind = set(ind[:maxActiveNodes])
#            tag_all, tag_1, tag_2 = [], [], []
#            sn = []
#            
#            # TODO: get rid of the cycles
#            for i in range(M):
#                cond1, cond2 = i in Ind, (i+M) in Ind
#                if cond1:
#                    if cond2:
#                        tag_all.append(i)
#                    else:
#                        tag_1.append(i)
#                else:
#                    if cond2:
#                        tag_2.append(i)
#                    else:
#                        sn.append(activeNodesCandidates[i])
#
#            list_lx, list_ux = [], []
#            
#            semiinvariant_new = []
#            updateTC = activeNodesCandidates[0].indT is not None
#            isRaw = p.solver.dataHandling == 'raw'
#            for i in tag_1:
#                node = activeNodesCandidates[i]
#                I = bestCoordsForSplitting[i]
##                if node.Lf[n+I] >= node.Lf[I]:
##                    print '1'
##                else:
##                    print i, I, node.Lf[n+I] ,  node.Lf[I], node.key, node.Uf[n+I] ,  node.Uf[I], node.nlhc[n+I], node.nlhc[I]
#                node.key = node.Lf[n+I]
#                node.semiinvariant = semiinvariant[i]
#                
#                if isRaw:
#                    node.tnlh_curr[I] = node.tnlh_curr[n+I]
#                    node.tnlh_curr_best = nanmin(node.tnlh_curr)
#                
#                #assert node.Lf[n+I] >= node.Lf[I]
#                #lx, ux = node.Lx, node.Ux
#                lx, ux = Lxc[i], Uxc[i]
#                if nn:
#                    #node.Lf[I], node.Uf[I] = node.Lf[n+I], node.Uf[n+I]
#                    node.Lf[I], node.Uf[I] = node.Lf[n+I], node.Uf[n+I]
#                    node.Lf[node.Lf<node.Lf[n+I]], node.Uf[node.Uf>node.Uf[n+I]] = node.Lf[n+I], node.Uf[n+I]
#                else:
#                    node.Lf[n+I], node.Uf[n+I] = node.Lf[I], node.Uf[I]
#                    node.Lf[node.Lf<node.Lf[I]], node.Uf[node.Uf>node.Uf[I]] = node.Lf[I], node.Uf[I]
#
##                if p.solver.dataHandling == 'raw':
#                for Attr in ('nlh_obj_fixed','nlhc', 'tnlh_fixed', 'tnlh_curr', 'tnlh_all'):
#                    r = getattr(node, Attr, None)
#                    if r is not None:
#                        if nn: r[I] = r[n+I]
#                        else: 
#                            r[n+I] = r[I]
#
#                mx = ux.copy()
#                mx[I] = Midles[i]#0.5*(lx[I] + ux[I])
#                list_lx.append(lx)
#                list_ux.append(mx)
#                node.Lx = lx.copy()
#                node.Lx[I] = Midles[i]#0.5*(lx[I] + ux[I])
#                if updateTC: 
#                    node.indT = True
#                
#                semiinvariant_new.append(node.semiinvariant)
#                sn.append(node)
#            
#            for i in tag_2:
#                node = activeNodesCandidates[i]
#                I = bestCoordsForSplitting[i]
#                node.key = node.Lf[I]
#
#                
#                node.semiinvariant = semiinvariant[i]
#                
#                # for raw only
#                if isRaw:
#                    node.tnlh_curr[n+I] = node.tnlh_curr[I]
#                    node.tnlh_curr_best = nanmin(node.tnlh_curr)
#                
#                #assert node.Lf[I] >= node.Lf[n+I]
#                #lx, ux = node.Lx, node.Ux
#                lx, ux = Lxc[i], Uxc[i]
#
#                if nn:
#                    node.Lf[n+I], node.Uf[n+I] = node.Lf[I], node.Uf[I]
#                    node.Lf[node.Lf<node.Lf[I]], node.Uf[node.Uf>node.Uf[I]] = node.Lf[I], node.Uf[I]
#                else:
#                    node.Lf[I], node.Uf[I] = node.Lf[n+I], node.Uf[n+I]
#                    node.Lf[node.Lf<node.Lf[n+I]], node.Uf[node.Uf>node.Uf[n+I]] = node.Lf[n+I], node.Uf[n+I]
#                for Attr in ('nlh_obj_fixed','nlhc', 'tnlh_fixed', 'tnlh_curr', 'tnlh_all'):
#                    r = getattr(node, Attr, None)
#                    if r is not None:
#                        if nn: r[n+I] = r[I]
#                        else: 
#                            r[I] = r[n+I]
#
#                mx = lx.copy()
#                mx[I] = Midles[i]#0.5*(lx[I] + ux[I])
#                list_lx.append(mx)
#                list_ux.append(ux)
#                node.Ux = ux.copy()
#                node.Ux[I] = Midles[i]#0.5*(lx[I] + ux[I])
#                if updateTC: 
#                    node.indT = True
#
#                semiinvariant_new.append(node.semiinvariant)
#                sn.append(node)
#            
#            for i in tag_all:
#                node = activeNodesCandidates[i]
#                I = bestCoordsForSplitting[i]
#                
#                #lx, ux = node.Lx, node.Ux
#                lx, ux = Lxc[i], Uxc[i]
#                mx = ux.copy()
#                mx[I] = Midles[i]#0.5 * (lx[I] + ux[I])
#                
#                list_lx.append(lx)
#                list_ux.append(mx)
#                
#                mx = lx.copy()
#                mx[I] = Midles[i]#0.5 * (lx[I] + ux[I])
#                #mx[n+ bestCoordsForSplitting] = 0.5 * (lx[n + bestCoordsForSplitting] + ux[n + bestCoordsForSplitting])
#                list_lx.append(mx)
#                list_ux.append(ux)
#                
#                #semiinvariant_new += [semiinvariant[i]] * 2
#                semiinvariant_new.append(semiinvariant[i])
#                semiinvariant_new.append(semiinvariant[i])
#                
##            print 'Lx_new:', vstack(list_lx)
##            print 'Ux_new:', vstack(list_ux)
##            print 'semiinvariant_new:', hstack(semiinvariant)
#            inactiveNodes = sn + inactiveNodes.tolist()
#            if p.solver.dataHandling == 'sorted':
#                inactiveNodes.sort(key = lambda obj: obj.key)
#            else:
#                #pass
#                inactiveNodes.sort(key = lambda obj: obj.tnlh_curr_best)
##            print 'tag 1:', len(tag_1), 'tag 2:', len(tag_2), 'tag all:', len(tag_all)
##            print 'lx:', list_lx
##            print 'sn lx:', [node.Lx for node in sn]
##            print 'ux:', list_ux
##            print 'sn ux:', [node.Ux for node in sn]
##            print '-'*10
#            #print '!', vstack(list_lx), vstack(list_ux), hstack(semiinvariant_new)
#            NEW_lx, NEW_ux, NEW_inactiveNodes, NEW_semiinvariant = \
#            vstack(list_lx), vstack(list_ux), inactiveNodes, hstack(semiinvariant_new)
#            return NEW_lx, NEW_ux, NEW_inactiveNodes, NEW_semiinvariant
