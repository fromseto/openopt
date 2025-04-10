PythonSum = sum
PythonMax = max
from numpy import tile, isnan, array, atleast_1d, asarray, logical_and, all, logical_or, any, nan, isinf, \
arange, vstack, inf, logical_not, take, abs, hstack, empty, \
prod, int16, int32, int64, log2, searchsorted, cumprod#where

# for PyPy
from openopt.kernel.nonOptMisc import where, isPyPy
from bisect import bisect, bisect_left, bisect_right

import numpy as np
from FuncDesigner.ooPoint import ooPoint
#from FuncDesigner import oopoint - doesn't work yet, incorrect class type check is gained

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
    
    
def func82(y, e, vv, f, dataType, p, Th = None, _s = None, indT = None):
    domain = ooPoint(((v, (y[:, i], e[:, i])) for i, v in enumerate(vv)), skipArrayCast=True, isMultiPoint=True)
    domain.dictOfFixedFuncs = p.dictOfFixedFuncs
    domain._dictOfRedirectedFuncs = p._dictOfRedirectedFuncs
    domain._dictOfStochVars = p._dictOfStochVars
    domain.maxDistributionSize = p.maxDistributionSize
    domain.nPoints = y.shape[0]
    domain._p = p
    
    new = 0
    if new:
        r, r0, y, e, _s, indT = f.iqg(domain, dataType, UB = Th, y = y, e = e, 
                              _s = _s, indT = indT)
    else:
#        r, r0, _y, _e  = f.iqg(domain, dataType, UB = Th)
        r, r0  = f.iqg(domain, dataType, UB = Th)
    if r is None:
        return None, None, None, None, False, False, array([]), array([])
    
    o_l, o_u, a_l, a_u = [], [], [], []
    definiteRange = r0.definiteRange
    for v in vv:
        # TODO: rework and optimize it
        tmp = r.get(v, None)
        if tmp is not None:
            o_l.append(tmp[0].lb)
            o_u.append(tmp[1].lb)
            a_l.append(tmp[0].ub)
            a_u.append(tmp[1].ub)
            
            #TODO: mb don't compute definiteRange for modificationVar != None
#            definiteRange = logical_and(definiteRange, tmp[0].definiteRange)
#            definiteRange = logical_and(definiteRange, tmp[1].definiteRange)
        else:
            o_l.append(r0.lb)
            o_u.append(r0.lb)
            a_l.append(r0.ub)
            a_u.append(r0.ub)
            #definiteRange = logical_and(definiteRange, r0.definiteRange)
    o, a = hstack(o_l+o_u), hstack(a_l+a_u)    
    return y, e, o, a, definiteRange, domain.exactRange, _s, indT

def func10(y, e, vv):
    m, n = y.shape
    LB = [[] for i in range(n)]
    UB = [[] for i in range(n)]

    r4 = (y + e) / 2
    
    for i in range(n):
        t1, t2 = tile(y[:, i], 2*n), tile(e[:, i], 2*n)
        t1[(n+i)*m:(n+i+1)*m] = t2[i*m:(i+1)*m] = r4[:, i]
        
#        if vv[i].domain is bool:
#            indINQ = y[:, i] != e[:, i]
#            tmp = t1[(n+i)*m:(n+i+1)*m]
#            tmp[indINQ] = 1
#            tmp = t2[i*m:(i+1)*m]
#            tmp[indINQ] = 0
            
#        if vv[i].domain is bool:
#            t1[(n+i)*m:(n+i+1)*m] = 1
#            t2[i*m:(i+1)*m] = 0
#        else:
#            t1[(n+i)*m:(n+i+1)*m] = t2[i*m:(i+1)*m] = r4[:, i]
        
        LB[i], UB[i] = t1, t2
    
    domain = dict((v, (LB[i], UB[i])) for i, v in enumerate(vv))
    domain = ooPoint(domain, skipArrayCast = True)
    domain.isMultiPoint = True
    domain.nPoints = 2*n*m#LB[0].shape[0]#2*n#4 * m#2*n#
    return domain

def func8(domain, func, dataType):
    TMP = func.interval(domain, dataType)
    #assert TMP.lb.dtype == dataType
    return asarray(TMP.lb, dtype=dataType), asarray(TMP.ub, dtype=dataType), TMP.definiteRange

def getr4Values(vv, y, e, activeCons, tnlh, func, C, contol, dataType, p, fo = inf):
    #TODO: use fo with objective before constraints when objective is simpler than constraints
    m, n = y.shape
    m0 = m
    wr4_0 = (y+e) / 2
    if tnlh is None:# or p.probType =='MOP':
        wr4 = wr4_0
    else:
        tnlh = tnlh.copy()
        tnlh[atleast_1d(tnlh<1e-300)] = 1e-300 # to prevent division by zero
        tnlh[atleast_1d(isnan(tnlh))] = inf #- check it!
        tnlh_l_inv, tnlh_u_inv = 1.0 / tnlh[:, :n], 1.0 / tnlh[:, n:]
        wr4 = (y * tnlh_l_inv + e * tnlh_u_inv) / (tnlh_l_inv + tnlh_u_inv)
        ind = tnlh_l_inv == tnlh_u_inv # especially important for tnlh_l_inv == tnlh_u_inv = 0
        wr4[ind] = (y[ind] + e[ind]) / 2
        wr4_1 = (wr4 + wr4_0)/2
        wr4 = vstack((wr4, wr4_0, wr4_1))
    # TODO: calculate cs in new way (median instead of rounding (lx + e) / 2)
    adjustr4WithDiscreteVariables(wr4, p)
    if p._isStochastic:
        cs = dict((oovar, asarray(wr4[:, i], dataType).reshape(-1, 1).view(multiarray)) for i, oovar in enumerate(vv))
    else:
        cs = dict((oovar, asarray(wr4[:, i], dataType)) for i, oovar in enumerate(vv))
        
    cs = ooPoint(cs, skipArrayCast = True, maxDistributionSize = p.maxDistributionSize)
    cs.isMultiPoint = True
    cs.dictOfFixedFuncs = p.dictOfFixedFuncs
    cs._dictOfRedirectedFuncs = p._dictOfRedirectedFuncs
    cs._dictOfStochVars = p._dictOfStochVars
    cs.nPoints = m
    
    m = wr4.shape[0]
    
    kw =  {'Vars': p.freeVars} if p.freeVars is None or (p.fixedVars is not None and len(p.freeVars) < len(p.fixedVars)) else {'fixedVars': p.fixedVars}
    kw['fixedVarsScheduleID'] = p._FDVarsID

    if len(C) != 0:
        r15 = empty(m, bool)
        r15.fill(True)
        for _c, f, r16, r17 in C.values():
            c = f(cs, **kw).flatten()
#            print(c,r16,r17)
            ind_feasible = logical_and(c  >= r16, c <= r17) # here r16 and r17 are already shifted by required tolerance
            r15 = logical_and(r15, ind_feasible)
            
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
        r15 = True
#    print('constr ind:',r15)
        
    isMOP = p.probType =='MOP'
    if not any(r15):
        F = empty(m, dataType)
        #TODO: replace 2**n by nan when integer nan in Python will be available
        F.fill(2**31-2 if dataType in (int16, int32, int64, int) else nan) # mb add int8
        if isMOP:
            FF = [F for k in range(p.nf)]
    elif all(r15):
        if isMOP:
            FF = [fun(cs, **kw) for fun in func]
        else:
            F = atleast_1d(asarray(func(cs, **kw), dataType).view(np.ndarray)).flatten() \
            if func is not None else np.zeros(m) # func can be None for SNLE, TODO: check is it still remain
    else:
        #cs = dict([(oovar, (y[r15, i] + e[r15, i])/2) for i, oovar in enumerate(vv)])
        #cs = ooPoint(cs, skipArrayCast = True)
        #cs.isMultiPoint = True
        if isMOP:
            FF = []
            for fun in func:
                tmp = atleast_1d(asarray(fun(cs, **kw), dataType).view(np.ndarray)).flatten() 
                F = empty(m, dataType)
                F.fill(2**31-15 if dataType in (int16, int32, int64, int) else nan)
                F[r15] = tmp[r15]    
                FF.append(F)
        else:
            tmp = atleast_1d(asarray(func(cs, **kw), dataType).view(np.ndarray)).flatten() if func is not None \
            else np.zeros(m) # func can be None for SNLE, TODO: check is it still remain
            
            F = empty(m, dataType)
            F.fill(2**31-15 if dataType in (int16, int32, int64, int) else nan)
            F[r15] = tmp[r15]
    #TODO: reduce cs by ind before computations
    if isMOP:
        return array(FF).view(np.ndarray).T.reshape(m, len(func)).tolist(), wr4.tolist()
    else:
        return F, wr4


def adjustr4WithDiscreteVariables(wr4, p):
    # TODO: rework it for discrete variables
    for i in p._discreteVarsNumList:
        v = p._freeVarsList[i]
        
        if v.domain is bool or v.domain is 'bool':
            wr4[:, i] = where(wr4[:, i]<0.5, 0, 1)
        else:
            tmp = wr4[:, i]
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
            wr4[:, i] = tmp
    #print where(wr4==0)[0].size, where(wr4==1)[0].size

def r2(PointVals, PointCoords, dataType):
#    r23 = nanargmin(PointVals)
#    if isnan(r23):
    # for bottleneck v >= 0.8:
    r23 = 0 if np.all(isnan(PointVals)) else nanargmin(PointVals)
    # TODO: check it , maybe it can be improved
    #bestCenter = cs[r23]
    #r7 = array([(val[0][r23]+val[1][r23]) / 2 for val in domain.values()], dtype=dataType)
    #r8 = atleast_1d(r3)[r23] if not isnan(r23) else inf
    r7 = array(PointCoords[r23], dtype=dataType)
    r8 = atleast_1d(PointVals)[r23] 
    return r7, r8
    
def func3(an, maxActiveNodes, p):

#    dataHandling = p.solver.dataHandling
    minActiveNodes = p.solver.minActiveNodes
    m = len(an)
    if m <= maxActiveNodes:
        return an, []#array([], object)
    
    an1, _in = an[:maxActiveNodes], an[maxActiveNodes:]    
        
    if getattr(an1[0], 'tnlh_curr_best', None) is not None:
        #t0 = an1[0].tnlh_curr_best
        tnlh_curr_best_values = asarray([node.tnlh_curr_best for node in an1])
        
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
        
        tmp1, tmp2 = an1[:M], an1[M:]
        an1 = tmp1
        _in = tmp2 + _in#hstack((tmp2, _in))
        
    # TODO: implement it for MOP as well
#    cond_min_uf = 0 and dataHandling == 'raw' and hasattr(an[0], 'key')            
#    
#    if cond_min_uf:
#        num_nlh = min((max((1, int(0.8*maxActiveNodes))), len(an1)))
#        num_uf = min((maxActiveNodes - num_nlh, int(maxActiveNodes/2)))
#        if num_uf < 15:
#            num_uf = 15
#        #an1, _in = an[:num_nlh], an[num_nlh:]    
#        Ind = np.argsort([node.key for node in _in])
#        
#        min_uf_nodes = _in[Ind[:num_uf]]
#        _in = _in[Ind[num_uf:]]
#        
#        an1 = an1 + min_uf_node#np.hstack((an1, min_uf_nodes))
        
    # changes end
    #print maxActiveNodes, len(an1), len(_in)
    
    return an1, _in

def func1(
        tnlhf_curr, nlhc, nlh_obj_fixed, residual, 
        y, e, o, a, _s_prev, p, indT, activeCons):
    
    m, n = y.shape
    w = arange(m)
    
    if p.probType == 'IP':
        oc_modL, oc_modU = o[:, :n], o[:, n:]
        ac_modL, ac_modU = a[:, :n], a[:, n:]
#            # TODO: handle nans
        mino = where(oc_modL < oc_modU, oc_modL, oc_modU)
        maxa = where(ac_modL < ac_modU, ac_modU, ac_modL)
    
        # Prev
        tmp = a[:, 0:n]-o[:, 0:n]+a[:, n:]-o[:, n:]
        t = nanargmin(tmp,1)
        d = 0.5*tmp[w, t]
        
        
        #New
#        tmp = a - o
#        t_ = nanargmin(tmp,1)
#        t = t_% n
#        d = tmp[w, t_]

#        ind = 2**(-n) >= (_s_prev - d)/asarray(d, 'float64')
        ind_problematic = 2**(1.0/n) * d >= _s_prev
        #new
#        ind = 2**(1.0/n) * d >= nanmax(maxa-mino, 1)
        
        #ind = 2**(-n) >= (_s_prev - _s)/asarray(_s, 'float64')
    
        #s2 = nanmin(maxa - mino, 1)
        #print (abs(s2/_s))
        
        # Prev
        _s = nanmin(maxa - mino, 1)
        
        # New
        #_s = nanmax(maxa - mino, 1)
#        _s = nanmax(a - o, 1)
        
        #ind = _s_prev  <= _s + ((2**-n / log(2)) if n > 15 else log2(1+2**-n)) 
        indD = None
        #print len(where(indD)[0]), len(where(logical_not(indD))[0])
#    elif p.probType == 'MOP':
#
#        raise 'unimplemented'
    else:
        if p.solver.dataHandling == 'sorted':
            _s = func13(o, a)
            
            t = nanargmin(a, 1) % n

            d = nanmax([a[w, t] - o[w, t], 
                    a[w, n+t] - o[w, n+t]], 0)
            
            ## !!!! Don't replace it by (_s_prev /d- 1) to omit rounding errors ###
            #ind = 2**(-n) >= (_s_prev - d)/asarray(d, 'float64')
            
            #NEW
            ind_problematic = d  >=  _s_prev / 2 ** (1.0e-1 / n)
            #ind = d  >=  _s_prev / 2 ** (1.0/n)
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
#                t = p._t[:m]
#                p._t = p._t[m:]
#                d = _s = p.__s[:m]
#                p.__s = p.__s[m:]
#                ind_problematic = _s_prev  <= d + delta
#            else:
            
            if p.probType == 'MOP':
                t = p._t[:m]
                p._t = p._t[m:]                    
            else:
                T = tnlhf_curr
                tnlh_curr_1, tnlh_curr_2 = T[:, :n], T[:, n:]
                TNHL_curr_min = where(logical_or(tnlh_curr_1 < tnlh_curr_2, isnan(tnlh_curr_2)), tnlh_curr_1, tnlh_curr_2)
                t = nanargmin(TNHL_curr_min, 1)
            
            # 1. Processing objective
            T = nlh_obj_fixed
            s_obj = nanmin(vstack((T[w, t], T[w, n+t])), 0)
            
#            if p.probType != 'ODE':
#                # TODO: mb use min/max instead of nanmin/nanmax
#                maxa, mino = nanmax(a, axis=1), nanmin(o, axis=1) #TODO: mb improve it (computate it wrt modL/modU)
#                ind_insufficient_obj_diff = maxa - mino < p.fTol
            
            # 2. Processing constraints
            #TODO: SKIP nlhc IF ALL ELEMENr38 ARE _constraintInactiveValue (OR mb if activeCons are empty lists)
            T = nlhc
            s_c = nanmin(vstack((T[w, t], T[w, n+t])), 0)
            
            # 3. Gathering results
            s_obj_prev, s_c_prev = np.real(_s_prev), np.imag(_s_prev)
            ind_problematic_c = logical_and(s_c_prev  <= s_c + delta, s_c != p.solver._constraintInactiveValue)
            
            ind_problematic_obj = s_obj_prev  <= s_obj + delta
            ind_problematic = logical_and(ind_problematic_obj, ind_problematic_c)
            _s = np.atleast_1d(s_obj+ 1j * s_c)
            
            #OLD
            #!#!#!#! Don't replace it by _s_prev - d <= ... to omit inf-inf = nan !#!#!#
            #ind = _s_prev  <= d + ((2**-n / log(2)) if n > 15 else log2(1+2**-n)) 
            #ind = _s_prev - d <= ((2**-n / log(2)) if n > 15 else log2(1+2**-n)) 
            
            #NEW
            indD = logical_or(ind_problematic, logical_not(indT))
#            indQ = d >= _s_prev - delta
#            indD = logical_or(indQ, logical_not(indT))
        else:
            assert 0

    if any(ind_problematic):
        r10_problematic = where(ind_problematic)[0]
        bs = e[r10_problematic] - y[r10_problematic]
        t[r10_problematic] = np.argmax(bs, 1) 
        
        # TODO: mb handle IP and ODE here
        skipProbs = ('IP', 'ODE')
        # TODO: mb handle case p.solver.dataHandling == 'sorted' here as well
        if 1 and p.probType not in skipProbs and p.solver.dataHandling == 'raw':
            ind_ignore_obj = logical_not(ind_problematic_obj[r10_problematic])
            
            if p.probType != 'MOP':
                aor20 = a[r10_problematic] - o[r10_problematic]
                
                # TODO: mb use the code below instead of mere nanmax(aor20)?
#            diff_mod_l, diff_mod_u = aor20[:, :n], aor20[:, n:]
#            max_ao_diff = where(diff_mod_l > diff_mod_u, diff_mod_l, diff_mod_u) # TODO: mb check for NaNs here
#            Max_ao_diff = np.max(max_ao_diff, axis=1)
                Max_ao_diff = np.nanmax(aor20, axis=1)#TODO: mb use max instead of nanmax (faster but less stable wrt possible future changes)
                
                r36mall_delta = Max_ao_diff < p.fTol # TODO: mb handle rTol here as well
                ind_ignore_obj |= r36mall_delta
                #  TODO: mb rework it, now it's just ind_ignore_obj = r36mall_delta
                
                
            #TODO: mb it can be reduced via skipping r36mall_delta processing

            #can be vectorized, but difference is insufficient
            constraints_dep_dict = p._constraints_dep_dict
            freeVarsDict = p._freeVarsDict
            objdep = p._objDep
            
            for j, i in enumerate(r10_problematic):
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
                    p.debugmsg('!!! Warning! len(dep) is zero in interalgLLR.func1()')
                    continue 
                
                # TODO: speedup inds_active obtaining
                inds_active = array([freeVarsDict[v] for v in dep])
                _ind = np.argmax(bs[j, inds_active])
                _i = inds_active[_ind]
                t[i] = _i
    
    return t, _s, indD
    
def func13(o, a): 
    m, n = o.shape
    n //= 2
#    if case == 1:
#        U1, U2 = a[:, :n].copy(), a[:, n:] 
#        #TODO: mb use nanmax(concatenate((U1,U2),3),3) instead?
#        U1 = where(logical_or(U1<U2, isnan(U1)),  U2, U1)
#        return nanmin(U1, 1)
        
    L1, L2, U1, U2 = o[:, :n], o[:, n:], a[:, :n], a[:, n:] 
#    if case == 2:
    U = where(logical_or(U1<U2, isnan(U1)),  U2, U1)
    L = where(logical_or(L2<L1, isnan(L1)), L2, L1)
    return nanmax(U-L, axis=1)

def func2(y, e, t, vv):#, tnlh_fixed_curr):
    #assert y.size != 0
    if y.size == 0: return y.copy(), e.copy() # especially for PyPy
    new_y, new_e = y.copy(), e.copy()
    m, n = y.shape
    w = arange(m)
    
    #!!!! TODO: rework it for discrete problems    
    th = (new_y[w, t] + new_e[w, t]) / 2
    
    ### !!!!!!!!!!!!!!!!!!!!!
    # TODO: rework it, don't recalculate each time
    DiscreteVars =  [v.domain is not None for v in vv] 
    BoolVars = [(v.domain is bool or v.domain is 'bool') for v in vv]

    indDiscrete = where(DiscreteVars)[0]
    discreteCoordsSet = set(indDiscrete)
    
    if 1 and all(BoolVars):
        # TODO: use it where y != e
        ind = y[w, t] != e[w, t]
        if any(ind):
            w2, t2 = w[ind], t[ind]
            new_y[w2, t2] = 1
            new_e[w2, t2] = 0
    
    else:
        new_y[w, t] = th
        new_e[w, t] = th
        
        # TODO: mb rework it
        for ind, coord in enumerate(t):
            if coord in discreteCoordsSet:
                #d = vv[coord]
                # = (0, 1) if d is bool or d is 'bool' else\
                m1, m2 = \
                splitDomainForDiscreteVariable(array([y[ind, coord]]), array([e[ind, coord]]), vv[coord])
                
                new_y[ind, coord] = m2
                new_e[ind, coord] = m1

    new_y = vstack((y, new_y))
    new_e = vstack((new_e, e))

    
#    if tnlhf_curr is not None:
#        tnlhf_curr_local = hstack((tnlhf_curr[w, t], tnlhf_curr[w, n+t]))
#    else:
#        tnlhf_curr_local = None
    return new_y, new_e#, tnlhf_curr_local


def func12(an, maxActiveNodes, p, Solutions, vv, varTols, fo):
    solutions, r6 = Solutions.solutions, Solutions.coords

    if len(an) == 0:
        return array([]), array([]), [], array([]), []
    _in = an
    
    if r6.size != 0:
        r11, r12 = r6 - varTols, r6 + varTols
    y, e, S, AC = [], [], [], []
    
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
#        return y, e, _in, _s
        
    
    while True:
        an1Candidates, _in = func3(_in, maxActiveNodes, p)

        #print nanmax(2**(-an1Candidates[0].tnlh_curr)) ,  nanmax(2**(-an1Candidates[-1].tnlh_curr))
        yc, ec, oc, ac, SIc = asarray([t.y for t in an1Candidates]), \
        asarray([t.e for t in an1Candidates]), \
        asarray([t.o for t in an1Candidates]), \
        asarray([t.a for t in an1Candidates]), \
        asarray([t._s for t in an1Candidates])
        
        activeCons = [t.activeCons for t in an1Candidates]
        
        if p.probType == 'MOP':
            
            # 
            # !!!!!!!! TODO: CHECK  tnlhf_curr <- tnlh_all 
            # VS tnlhf_curr <- tnlh_curr in next section
            #
            if p.solver.mop_mode == 1:
                tnlhf_curr = asarray([t.tnlh_all for t in an1Candidates])
            else:
                tnlhf_curr = None
            nlhc = asarray([t.nlhc for t in an1Candidates]) 
#            nlh_obj_fixed = None
            nlh_obj_fixed = asarray([t.nlh_obj_fixed for t in an1Candidates]) 
        elif p.solver.dataHandling == 'raw':
            tnlhf_curr = asarray([t.tnlh_curr for t in an1Candidates]) 
            nlhc = asarray([t.nlhc for t in an1Candidates]) 
            nlh_obj_fixed = asarray([t.nlh_obj_fixed for t in an1Candidates]) 
        else:
            tnlhf_curr, nlhc, nlh_obj_fixed = None, None, None
        
        
        if p.probType != 'IP': 
            #nlhc = asarray([t.nlhc for t in an1Candidates])
            
            #residual = asarray([t.residual for t in an1Candidates]) 
            residual = None
            
            # TODO: implement it in interalgMOP code
#            if p.probType != 'MOP':
            if p.solver.mop_mode == 1:
                indT = func4(p, yc, ec, oc, ac, fo, tnlhf_curr)
                if an1Candidates[0].indtc is not None:
                    indtc = asarray([t.indtc for t in an1Candidates])
                    indT = logical_or(indT, indtc)
            else:
                indT = asarray([t.indtc for t in an1Candidates])
        else:
            residual = None
            indT = None
        t, _s, indD = func1(tnlhf_curr, nlhc, nlh_obj_fixed, residual, yc, ec, oc, ac, SIc, p, indT, activeCons)
        
        NewD = 1
        
        if NewD and indD is not None:#TODO: mb handle case  all(indD) here and skip to part 2
            ind_divide, ind_remain = where(indD)[0], where(logical_not(indD))[0]
            s4d = _s[ind_divide]
            sf = _s[ind_remain]
            AC_divide, AC_remain = [activeCons[i] for i in ind_divide],\
            [activeCons[i] for i in ind_remain]

            _s = hstack((s4d, s4d, sf))
            
            #activeCons = 2*AC_divide + AC_remain
            activeCons = AC_divide+[elem.copy() for elem in AC_divide] + AC_remain
            
            yf, ef = yc[ind_remain], ec[ind_remain]
            yc, ec = yc[ind_divide], ec[ind_divide]
            t = t[ind_divide]
            
        else:
            _s = tile(_s, 2)
            #activeCons = 2*activeCons # Python list duplication
            activeCons = activeCons + [elem.copy() for elem in activeCons]

        #yc, ec, tnlhf_curr_local = func2(yc, ec, t, vv, tnlhf_curr)
        yc, ec = func2(yc, ec, t, vv)

        if NewD and indD is not None:
            yc = vstack((yc, yf))
            ec = vstack((ec, ef))
            
        if maxSolutions == 1 or len(solutions) == 0: 
            #y, e, Tnlhf_curr_local = yc, ec, tnlhf_curr_local
            y, e = yc, ec
            break
        
        # TODO: change cycle variable if len(solutions) >> maxActiveNodes
        for i in range(len(solutions)):
            ind = logical_and(all(yc >= r11[i], 1), all(ec <= r12[i], 1))
            if any(ind):
                j = where(logical_not(ind))[0]
                lj = j.size
                yc = take(yc, j, axis=0, out=yc[:lj])
                ec = take(ec, j, axis=0, out=ec[:lj])
                _s = _s[j]
                
#                for i in where(ind)[0]:
#                    box_active_constraints = activeCons[i]
#                    for c_id in box_active_constraints:
#                        constraints_counter[c_id] -= 1 
                activeCons = [activeCons[idx] for idx in j]
                
#                if tnlhf_curr_local is not None:
#                    tnlhf_curr_local = tnlhf_curr_local[j]
        y.append(yc)
        e.append(ec)
        S.append(_s)
        AC.append(activeCons)
        #Tnlhf_curr_local.append(tnlhf_curr_local)
        N += yc.shape[0]
        if len(_in) == 0 or N >= maxActiveNodes: 
            y, e, _s = vstack(y), vstack(e), hstack(S)
            activeCons = PythonSum(AC, []) 
            #Tnlhf_curr_local = hstack(Tnlhf_curr_local)
            break
    
    assert len(activeCons) == _s.size
    
    return y, e, _in, _s, activeCons

Fields = ['key', 'y', 'e', 'nlh_obj_fixed','nlhc', 'indtc','residual', 'o', 'a', \
'_s', 'activeCons']
MOP_Fields = Fields[1:]
IP_fields = ['key', 'y', 'e', 'o', 'a', '_s', 'F', 'volume', 'volumeResidual', 'activeCons']

# TODO: add activeCons when cons will be implemented for IP

def func11(y, e, nlhc, indT, residual, o, a, _s, p, activeCons, ind_nan = None): 
    y, e, o, a, _s, indT, nlhc, residual, activeCons = \
    func7(y, e, o, a, _s, indT, nlhc, residual, activeCons, p._constraints_counter, ind_nan)
    
    m, n = y.shape
    if m == 0:
        assert p.probType not in ('IP', 'ODE'), 'unimplemented for functions with domain including NaNs yet'
        return [], y, e, o, a, _s, indT, nlhc, residual, activeCons
        
    assert m == _s.size == len(activeCons)
    if p.probType == "IP":
        w = arange(m)
        # TODO: omit recalculation from func1
        ind = nanargmin(a[:, 0:n] - o[:, 0:n] + a[:, n:] - o[:, n:], 1)
        sup_inf_diff = 0.5*(a[w, ind] - o[w, ind] + a[w, n+ind] - o[w, n+ind])
        volume = prod(e-y, 1)
        volumeResidual = volume * sup_inf_diff
        F = (a[w, ind] + o[w, ind] + a[w, n+ind] + o[w, n+ind]) / 4.0
        
        return [si(p, IP_fields, sup_inf_diff[i], y[i], e[i], o[i], a[i], _s[i], 
        F[i], volume[i], volumeResidual[i], activeCons[i]) for i in range(m)], y, e, o, a, _s, indT, nlhc, residual, activeCons
        

    residual = None
    
    isSNLE = p.probType in ('SNLE', 'NLSP')
    if 1 or not isSNLE:
        o, a = asarray(o), asarray(a)
        
        ind_inf = a==inf
        if ind_inf.any(): # without this check doesn't work with integer types
            a[ind_inf] = 1e300
        ind_inf = o==inf
        if ind_inf.any(): # without this check doesn't work with integer types
            o[o==-inf] = -1e300
        tmp = a - o
        
        
        #changes!!
        tmp[tmp<1e-300] = 1e-300
        #tmp[tmp<p.fTol] = p.fTol # TODO: mb rework it, e.g. replace by 1e-300
        
        
        
#            ind_uf_inf = where(a==inf)[0]
#            if ind_uf_inf.size:
#                Tmp = o[ind_uf_inf]
#                Tmp[Tmp==-inf] = -1e100
#                M = nanmax(abs(Tmp))
#                if M is nan or M == 0.0: 
#                    M = 1.0
#                tmp[ind_uf_inf] = 1e200 * (1.0 + Tmp/M)
        nlh_obj_fixed = log2(tmp)#-log2(p.fTol)
        
#        nlh_obj_fixed[a==inf] = 1e300# to make it not inf and nan
#        nlh_obj_fixed[o==-inf] = 1e300# to make it not inf and nan
    
    z = np.empty(2*n) # TODO: use None instead (mb replace by vector while encoding)
    z.fill(p.solver._constraintInactiveValue)# ~= log2(1e-300)
    
    if p.probType == "MOP":
        assert nlh_obj_fixed.ndim == 3, 'bug in interalg' 
        nlh_obj_fixed = nlh_obj_fixed.sum(axis=1)
        return [si(
                        p, MOP_Fields, y[i], e[i], nlh_obj_fixed[i], 
                        nlhc[i] if nlhc is not None else z, 
                        indT[i] if indT is not None else True, 
                        residual[i] if residual is not None else None, 
                        [o[i][k] for k in range(p.nf)], 
                        [a[i][k] for k in range(p.nf)], 
                        _s[i], 
                        activeCons[i] #if activeCons is not None else None
                        ) for i in range(m)], y, e, o, a, _s, indT, nlhc, residual, activeCons

    assert p.probType in ('GLP', 'NLP', 'NSP', 'SNLE', 'NLSP', 'MINLP', 'QP', 'LP', 'MILP')
    
    if 0 and isSNLE:
        nlh_obj_fixed = Tmp = o = a = [None]*m
    else:
        s, q = o[:, :n], o[:, n:]
        Tmp = nanmax(where(q<s, q, s), 1)
#        print('y:', y)
#        print('e:', e)
#        print('o:', o)
#        print('Tmp:', Tmp)
        # TODO: mb rework it
        nlh_obj_fixed[logical_and(isinf(a), isinf(nlh_obj_fixed))] = 1e300
    
#            residual = None

    return [si(p, Fields, Tmp[i], y[i], e[i], nlh_obj_fixed[i], 
                  nlhc[i] if nlhc is not None else z, 
                  indT[i] if indT is not None else None, 
                  residual[i] if residual is not None else None, 
                  o[i], a[i], _s[i], 
                  activeCons[i] #if activeCons is not None else None
                  ) for i in range(m)], y, e, o, a, _s, indT, nlhc, residual, activeCons


class si:
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
#            arr = tnlhf_curr if p.solver.dataHandling == 'raw' else oc
#            M = arr.shape[0]
#            w = arange(M)
#            Midles = 0.5*(yc[w, t] + ec[w, t])
#            arr_1, arr2 = arr[w, t], arr[w, n+t]
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
#                        sn.append(an1Candidates[i])
#
#            list_lx, list_ux = [], []
#            
#            _s_new = []
#            updateTC = an1Candidates[0].indT is not None
#            isRaw = p.solver.dataHandling == 'raw'
#            for i in tag_1:
#                node = an1Candidates[i]
#                I = t[i]
##                if node.o[n+I] >= node.o[I]:
##                    print '1'
##                else:
##                    print i, I, node.o[n+I] ,  node.o[I], node.key, node.a[n+I] ,  node.a[I], node.nlhc[n+I], node.nlhc[I]
#                node.key = node.o[n+I]
#                node._s = _s[i]
#                
#                if isRaw:
#                    node.tnlh_curr[I] = node.tnlh_curr[n+I]
#                    node.tnlh_curr_best = nanmin(node.tnlh_curr)
#                
#                #assert node.o[n+I] >= node.o[I]
#                #lx, ux = node.y, node.e
#                lx, ux = yc[i], ec[i]
#                if nn:
#                    #node.o[I], node.a[I] = node.o[n+I], node.a[n+I]
#                    node.o[I], node.a[I] = node.o[n+I], node.a[n+I]
#                    node.o[node.o<node.o[n+I]], node.a[node.a>node.a[n+I]] = node.o[n+I], node.a[n+I]
#                else:
#                    node.o[n+I], node.a[n+I] = node.o[I], node.a[I]
#                    node.o[node.o<node.o[I]], node.a[node.a>node.a[I]] = node.o[I], node.a[I]
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
#                node.y = lx.copy()
#                node.y[I] = Midles[i]#0.5*(lx[I] + ux[I])
#                if updateTC: 
#                    node.indT = True
#                
#                _s_new.append(node._s)
#                sn.append(node)
#            
#            for i in tag_2:
#                node = an1Candidates[i]
#                I = t[i]
#                node.key = node.o[I]
#
#                
#                node._s = _s[i]
#                
#                # for raw only
#                if isRaw:
#                    node.tnlh_curr[n+I] = node.tnlh_curr[I]
#                    node.tnlh_curr_best = nanmin(node.tnlh_curr)
#                
#                #assert node.o[I] >= node.o[n+I]
#                #lx, ux = node.y, node.e
#                lx, ux = yc[i], ec[i]
#
#                if nn:
#                    node.o[n+I], node.a[n+I] = node.o[I], node.a[I]
#                    node.o[node.o<node.o[I]], node.a[node.a>node.a[I]] = node.o[I], node.a[I]
#                else:
#                    node.o[I], node.a[I] = node.o[n+I], node.a[n+I]
#                    node.o[node.o<node.o[n+I]], node.a[node.a>node.a[n+I]] = node.o[n+I], node.a[n+I]
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
#                node.e = ux.copy()
#                node.e[I] = Midles[i]#0.5*(lx[I] + ux[I])
#                if updateTC: 
#                    node.indT = True
#
#                _s_new.append(node._s)
#                sn.append(node)
#            
#            for i in tag_all:
#                node = an1Candidates[i]
#                I = t[i]
#                
#                #lx, ux = node.y, node.e
#                lx, ux = yc[i], ec[i]
#                mx = ux.copy()
#                mx[I] = Midles[i]#0.5 * (lx[I] + ux[I])
#                
#                list_lx.append(lx)
#                list_ux.append(mx)
#                
#                mx = lx.copy()
#                mx[I] = Midles[i]#0.5 * (lx[I] + ux[I])
#                #mx[n+ t] = 0.5 * (lx[n + t] + ux[n + t])
#                list_lx.append(mx)
#                list_ux.append(ux)
#                
#                #_s_new += [_s[i]] * 2
#                _s_new.append(_s[i])
#                _s_new.append(_s[i])
#                
##            print 'y_new:', vstack(list_lx)
##            print 'e_new:', vstack(list_ux)
##            print '_s_new:', hstack(_s)
#            _in = sn + _in.tolist()
#            if p.solver.dataHandling == 'sorted':
#                _in.sort(key = lambda obj: obj.key)
#            else:
#                #pass
#                _in.sort(key = lambda obj: obj.tnlh_curr_best)
##            print 'tag 1:', len(tag_1), 'tag 2:', len(tag_2), 'tag all:', len(tag_all)
##            print 'lx:', list_lx
##            print 'sn lx:', [node.y for node in sn]
##            print 'ux:', list_ux
##            print 'sn ux:', [node.e for node in sn]
##            print '-'*10
#            #print '!', vstack(list_lx), vstack(list_ux), hstack(_s_new)
#            NEW_lx, NEW_ux, NEW__in, NEW__s = \
#            vstack(list_lx), vstack(list_ux), _in, hstack(_s_new)
#            return NEW_lx, NEW_ux, NEW__in, NEW__s
