PythonAny = any
PythonAll = all
from numpy import isnan, take, any, all, logical_or, logical_and, logical_not, atleast_1d, \
argmin, argsort, abs, isfinite
import numpy as np

# for PyPy
from openopt.kernel.nonOptMisc import where

from bisect import bisect_right
from FuncDesigner.Interval import splitDomainForDiscreteVariable

try:
    from bottleneck import nanmin
except ImportError:
    from numpy import nanmin

def getTruncatedArrays(ind, Lx, Ux, indT, semiinvariant):
    # TODO: rework it when numpy will have appropriate inplace function
    s = ind.size
    Lx = take(Lx, ind, axis=0, out=Lx[:s])
    Ux = take(Ux, ind, axis=0, out=Ux[:s])
    semiinvariant = semiinvariant[ind]
    if indT is not None:
        indT = indT[ind]
    return Lx, Ux, indT, semiinvariant#, nlh, nlh_0
    
def adjustDiscreteVarBounds(Lx, Ux, p):
    # TODO: rework it
    #n = p.n
    # TODO: remove the cycle, use vectorization
    for i in p._discreteVarsNumList:
        v = p._freeVarsList[i]
        Lx[:, i], Ux[:, i] = splitDomainForDiscreteVariable(Lx[:, i], Ux[:, i], v)

#    ind = Lx>Ux
#    assert not any(ind)
    
    
#    Lx[ind], Ux[ind] = Ux[ind], Lx[ind]

#    Ind = any(Lx>Ux, 1)
#    trunc_ind = where(logical_not(Ind))[0]

#    # TODO:  is it triggered? // updated: can be from MOP or cons
#    if any(Ind):
#        ind = where(logical_not(Ind))[0]
#        s = ind.size
#        Lx = take(Lx, ind, axis=0, out=Lx[:s])
#        Ux = take(Ux, ind, axis=0, out=Ux[:s])
#        semiinvariant = semiinvariant[ind]
#        if indT is not None:
#            indT = indT[ind]
    return Lx, Ux#, trunc_ind#semiinvariant, indT

    
def remove_NaN_nodes(Lx, Ux, Lf, Uf, semiinvariant, indT, nlhc, residual, activeCons, constraints_counter, ind_nan):
    
    assert len(activeCons) == semiinvariant.size == Lx.shape[0]
    
    isMOP = type(Lf) == list
    m = Lx.shape[0]
    if ind_nan is not None:# from MOP
        IND = ind_nan
#    elif isMOP:
#        
##        ind_nan_lf = []
#        lf_inds_with_nan = []
#        uf_inds_with_nan = []
##        for i in range(m-1, 0, -1):
#        
##        units = [np.vstack(elem) for k, elem in enumerate(Lf[i])) for i in range(m)]    
#        tmp = [[isnan(np.vstack(elem)) for k, elem in enumerate(Lf[i])] for i in range(m)]
#        tmp2 = [[np.vstack(elem) for k, elem in enumerate(Lf[i])] for i in range(m)]
#        
#        for i in range(m):
#            tmp = np.vstack(Lf[i])
#            has_nan = any(isnan(tmp), axis=0)
#            lf_inds_with_nan.append(has_nan)
#            print(tmp)
#        
#        
#        lf_inds_with_nan = [PythonAll(np.any(isnan(elem)) for k, elem in enumerate(Lf[i])) for i in range(m)]    
#        # TODO: mb skip Uf part, mb eats much cpu
#        uf_inds_with_nan = [PythonAll(np.any(isnan(elem)) for k, elem in enumerate(Uf[i])) for i in range(m)]
#        assert lf_inds_with_nan == uf_inds_with_nan, 'bug in interval analysis'
#        IND = np.array(lf_inds_with_nan)
#        assert 0, 'unimplemented yet'
    else:
        #print('interalgT: requires further improvements')
        
        IND = logical_and(all(isnan(Lf), axis=1), all(isnan(Uf), axis=1))
    
    #TODO: mb further improvements (use halving instead of processing whole box)
    
    j = where(logical_not(IND))[0]
    lj = j.size
        
    if lj != m:
        Lx = take(Lx, j, axis=0, out=Lx[:lj])
        Ux = take(Ux, j, axis=0, out=Ux[:lj])
        if isMOP:
            ind_remove = where(IND)[0][::-1]
            for i in ind_remove:
                Lf.pop(i)
                Uf.pop(i)
        else:
            Lf = take(Lf, j, axis=0, out=Lf[:lj])
            Uf = take(Uf, j, axis=0, out=Uf[:lj])
            
        semiinvariant = semiinvariant[j]
        if indT is not None:
            indT = indT[j]
        if nlhc is not None:
            nlhc = take(nlhc, j, axis=0, out=nlhc[:lj])
        if residual is not None:
            residual = take(residual, j, axis=0, out=residual[:lj])
        
#        if activeCons is not None:

        # excluding activeCons
        for i in where(IND)[0]:
            box_active_constraints = activeCons[i]
            for c_id in box_active_constraints:
                constraints_counter[c_id] -= 1 
#        l1 = len(activeCons)
        activeCons = [activeCons[i] for i in j]
#        print('assdf')
#        l2 = len(activeCons)
#        if l2 != l1:
#            print ('removed %d active cons' % (l1-l2))
    
    assert len(activeCons) == semiinvariant.size == Lx.shape[0]
    
    return Lx, Ux, Lf, Uf, semiinvariant, indT, nlhc, residual, activeCons
    

def removeSomeNodes(AllNodes, threshold, cutLevel, p):
    
    #ind = searchsorted(maxLf, threshold, side='right')
    if p.probType in ('NLSP', 'SNLE') and p.maxSolutions != 1:
        minLf = atleast_1d([node.key for node in AllNodes])
        ind = minLf > 0
        if not any(ind):
            return AllNodes, cutLevel

        cutLevel = nanmin((cutLevel, nanmin(minLf[ind])))
        ind2 = where(logical_not(ind))[0]
        #AllNodes = take(AllNodes, ind2, axis=0, out=AllNodes[:ind2.size])
        #AllNodes = asarray(AllNodes[ind2])
        AllNodes = [AllNodes[i] for i in ind2]
        return AllNodes, cutLevel
            
        
    elif p.solver.dataHandling == 'sorted':
        #OLD
        minLf = [node.key for node in AllNodes]
        ind = bisect_right(minLf, threshold)
        if ind == len(minLf):
            return AllNodes, cutLevel
        else:
            cutLevel = nanmin((cutLevel, nanmin(atleast_1d(minLf[ind]))))
            return AllNodes[:ind], cutLevel
    elif p.solver.dataHandling == 'raw':
        
        #NEW
        minLf = atleast_1d([node.key for node in AllNodes])
        IND = minLf > threshold
        
        if not any(IND):
            return AllNodes, cutLevel
            
        ind = where(IND)[0]
        cutLevel = nanmin((cutLevel, nanmin(atleast_1d(minLf)[ind])))
        
        if 1:
            ind2 = where(logical_not(IND))[0]
            AllNodes = [AllNodes[i] for i in ind2]
        else:
            #CURRENTLY WORKS SLOWER
            AllNodes = [AllNodes[j] for j in range(len(AllNodes)) if not(IND[j])]
        
        return AllNodes, cutLevel

        # NEW 2
#        curr_tnlh = [node.tnlh_curr for node in AllNodes]
#        import warnings
#        warnings.warn('! fix cutLevel')
        
        return AllNodes, cutLevel
        
    else:
        assert 0, 'incorrect nodes remove approach'

def TruncateOutOfAllowedNumberNodes(AllNodes, nCut, cutLevel, p):
    m = len(AllNodes)
    if m <= nCut: return AllNodes, cutLevel
    
    minLf = np.array([node.key for node in AllNodes])
    
    if nCut == 1: # box-bound probs with exact interval analysis
        ind = argmin(minLf)
        assert ind in (0, 1), 'error in interalg engine'
        cutLevel = nanmin((minLf[1-ind], cutLevel))
        AllNodes = [AllNodes[i] for i in ind]
    elif m > nCut:
        if p.solver.dataHandling == 'raw':
            # TODO: mb add mode for truncation by tnlh instead of lb
            ind = argsort(minLf)
            th = minLf[ind[nCut]]
            ind2 = where(minLf < th)[0]
            cutLevel = nanmin((th, cutLevel))
            #AllNodes = take(AllNodes, ind2, axis=0, out=AllNodes[:ind2.size])
            AllNodes = [AllNodes[i] for i in ind2]#AllNodes[ind2]
        else:
            cutLevel = nanmin((minLf[nCut], cutLevel))
            AllNodes = AllNodes[:nCut]
    return AllNodes, cutLevel

def TruncateSomeBoxes(p, Lx, Ux, Lf, Uf, threshold, tnlhf_curr = None):
    # TODO: simplifications for all-bool probs
    if threshold is None and tnlhf_curr is None: return False# used in IP
    if Lx.size == 0: return False
    centers = (Lx + Ux)/2.0
    n = Lx.shape[1]
    # TODO: Uf, Lf  could be chenged to +/- inf instead of values duplication
    
    if tnlhf_curr is not None:
        tnlh_modL = tnlhf_curr[:, :n]
        # TODO: mb replace it by tnlh_modL == inf
        ind = logical_not(isfinite(tnlh_modL))
    else:
        Lf_modL = Lf[:, :n]
        ind = logical_or(Lf_modL > threshold, isnan(Lf_modL)) # TODO: assert isnan(Lf_modL) is same to isnan(Uf_modL)
#    hasDiscreteVariables = len(p._discreteVarsNumList) != 0
    indT = any(ind, 1)
    if any(ind):
        _ind = where(ind)
        #TODO: improve and finish replacement ind by _ind
        # and make it done for Ux as well
        
#        if hasDiscreteVariables:
        for j, v in enumerate(p._discreteVarsList):
            i = p._discreteVarsNumList[j]
            k = where(ind[:, i])[0]
            if k.size == 0: continue
            discr_mid1, discr_mid2 = splitDomainForDiscreteVariable(Lx[k, i], Ux[k, i], v)
            centers[k, i] = discr_mid2
        
        Lx[_ind] = centers[_ind]

        # TODO: implement if for MOP 
        if p.probType != 'MOP': 
#            s1 = Uf[:, :n][_ind]
            Uf[:, :n][_ind] = Uf[:, n:][_ind]
            #Uf[ind, :n] = Uf[:, n:][ind]
#            s2 = Uf[:, :n][_ind]
#            print (max(s2-s1), max(s1-s2))
            Lf[:, :n][_ind] = Lf[:, n:][_ind]
        if tnlhf_curr is not None:
            tnlhf_curr[:, :n][_ind] = tnlhf_curr[:, n:][_ind]


    if tnlhf_curr is not None:
        tnlh_modU = tnlhf_curr[:, n:]
        ind = logical_not(isfinite(tnlh_modU))
    else:
        Lf_modU = Lf[:, n:]
        ind = logical_or(Lf_modU > threshold, isnan(Lf_modU)) # TODO: assert isnan(Lf_modU) is same to isnan(Uf_modU)
        
    indT = logical_or(any(ind, 1), indT)
    if any(ind):
        # copy is used to prevent Lx and Ux being same array, that may be buggy with discrete vars
        # TODO: remove copy after new discrete vars handling implementation
        for j, v in enumerate(p._discreteVarsList):
            i = p._discreteVarsNumList[j]
            k = where(ind[:, i])[0]
            if k.size == 0: continue
            discr_mid1, discr_mid2 = splitDomainForDiscreteVariable(Lx[k, i], Ux[k, i], v)
            centers[k, i] = discr_mid1
            
        Ux[ind] = centers[ind].copy() 
        # Changes
#        ind = logical_and(ind, logical_not(isnan(Uf[:, n:])))
##        ii = len(where(ind)[0])
##        if ii != 0: print ii

        if p.probType != 'MOP':
            # TODO: implement if for MOP 
            Uf[:, n:][ind] = Uf[:, :n][ind]
            Lf[:, n:][ind] = Lf[:, :n][ind]
        if tnlhf_curr is not None:
            tnlhf_curr[:, n:][ind] = tnlhf_curr[:, :n][ind]
#        for arr in arrays:
#            if arr is not None:
#                arr[:, n:2*n][ind] = arr[:, 0:n][ind]
        
    return indT
    
def truncateByPlane(Lx, Ux, indT, A, b):
    #!!!!!!!!!!!!!!!!!!!
    # TODO: vectorize it by matrix A
    #!!!!!!!!!!!!!!!!!!!
    ind_remain = None
#    assert np.asarray(b).size <= 1, 'unimplemented yet'
    m, n = Lx.shape
    if m == 0:
        return Lx, Ux, indT, ind_remain

    ind_positive = where(A > 0)[0]
    ind_negative = where(A < 0)[0]
#    print ind_negative, ind_positive
    A1 = A[ind_positive] 
    S1 = A1 * Lx[:, ind_positive]
    A2 = A[ind_negative]
    S2 = A2 * Ux[:, ind_negative]
    
#    print A1.shape, A2.shape, Ux[:, ind_negative].shape
    S = S1.sum(axis=1) + S2.sum(axis=1)

    if ind_positive.size != 0:
        S1_ = b - S.reshape(-1, 1) + S1
        Alt_ub = S1_ / A1

        ind = Ux[:, ind_positive] > Alt_ub #+ 1e-15*abs(Alt_ub)
        if np.any(ind):
            _ind = where(ind.flatten())[0]
            Ux[:, ind_positive].flat[_ind] = Alt_ub.flat[_ind]
            indT[any(ind, axis = 1)] = True

    
    if ind_negative.size != 0:
        S2_ = b - S.reshape(-1, 1) + S2
        Alt_lb = S2_ / A2
        
        ind = Lx[:, ind_negative] < Alt_lb #- 1e-15 * abs(Alt_lb)
        if np.any(ind):
            #Lx[:, ind_negative][ind] = Alt_lb[ind]
            _ind = where(ind.flatten())[0]
            Lx[:, ind_negative].flat[_ind] = Alt_lb.flat[_ind]
            indT[any(ind, axis = 1)] = True


    ind = all(Ux>=Lx, axis = 1)
    if not all(ind):
        ind_remain = where(ind)[0]
        lj = ind_remain.size
        Lx = take(Lx, ind_remain, axis=0, out=Lx[:lj])
        Ux = take(Ux, ind_remain, axis=0, out=Ux[:lj])
        indT = indT[ind_remain]
            
    return Lx, Ux, indT, ind_remain


def _truncateByPlane(Lx, Ux, indT, A, b):
    #!!!!!!!!!!!!!!!!!!!
    # TODO: vectorize it by matrix A
    #!!!!!!!!!!!!!!!!!!!
    ind_remain = slice(None)
#    assert np.asarray(b).size <= 1, 'unimplemented yet'
    m, n = Lx.shape
    A = A.T
    assert A.shape == Lx.shape
    if m == 0:
        assert Ux.size == 0, 'bug in interalg engine'
        return Lx, Ux, indT, ind_remain

#    ind_positive = where(A > 0)[0]
#    ind_negative = where(A < 0)[0]
#    print ind_negative, ind_positive

    #TODO: mb store ind_positive, ind_negative in prob for fixed A

    ind_positive, ind_negative = where(A>0), where(A<0)
#    S1 = dot(where(A>0, A, -A).T, Lx.T)
#    S2 = dot(where(A<0, A, -A).T, Ux.T)
    
#    S_Lx = dot(where(A>0, A, 0).T, Lx.T)
#    S_Lx_ = dot(where(A<0, A, 0).T, Lx.T)
#    S_Ux = dot(where(A>0, A, 0).T, Ux.T)
#    S_Ux_ = dot(where(A<0, A, 0).T, Ux.T)

    S_Lx = where(A>0, A, 0) * Lx
    S_Lx_ = where(A<0, A, 0) * Lx
    S_Ux = where(A>0, A, 0) * Ux
    S_Ux_ = where(A<0, A, 0) * Ux
    

#    _S = S1+S2

    # 1
    S1 = S_Lx 
    S2 = S_Ux_
    S = S1.sum(axis=1) + S2.sum(axis=1)
    d = (b.reshape(-1, 1)  - S.reshape(-1, 1)) + S1 # vector + matrix
    Alt = (d / A) # vector / matrix
    #Alt = 
    
    ind = Ux[ind_positive] > Alt[ind_positive] + 1e-15*(abs(Alt[ind_positive])+abs(Ux[ind_positive]))
    
    if np.any(ind):
        _ind = where(ind.flatten())[0]
        Ux0 = Ux.copy()# TODO: mb rework it
        Ux[ind_positive].flat[_ind] = Alt.flat[_ind]
        ind_ = any(Ux!=Ux0, axis = 1)
#        indT[any(ind_, axis = 1)] = True
        indT[ind_] = True
    
    # 2
    S1 = S_Lx_
    S2 = S_Ux
    S = S1.sum(axis=1) + S2.sum(axis=1)
    
    d = (b.reshape(-1, 1)  - S.reshape(-1, 1)) + S2 # vector + matrix
    Alt = (d / A) # vector / matrix
    
    ind = Lx[ind_negative] < Alt[ind_negative] - 1e-15 * (abs(Alt[ind_negative]) + abs(Lx[ind_negative]))
    if np.any(ind):
        #Lx[:, ind_negative][ind] = Alt[ind]
        _ind = where(ind.flatten())[0]
        Lx0 = Lx.copy()# TODO: mb rework it
        Lx[ind_negative].flat[_ind] = Alt.flat[_ind]
        ind_ = any(Lx!=Lx0, axis = 1)
#        indT[any(ind_, axis = 1)] = True
        indT[ind_] = True
    # 3

    ind = all(Ux>=Lx, axis = 1)
    if not all(ind):
        ind_remain = where(ind)[0]
        lj = ind_remain.size
        Lx = take(Lx, ind_remain, axis=0, out=Lx[:lj])
        Ux = take(Ux, ind_remain, axis=0, out=Ux[:lj])
        indT = indT[ind_remain]
    
    return Lx, Ux, indT, ind_remain
    
def truncateByPlane2(centers, centerValues, Lx, Ux, indT, gradient, threshold, p):
    
    ind_remain = True
    assert np.asarray(threshold).size <= 1, 'unimplemented yet'
    m, n = Lx.shape
    if m == 0:
        assert Ux.size == 0, 'bug in interalg engine'
        return Lx, Ux, indT, ind_remain

    oovarsIndDict = p._oovarsIndDict
    ind = np.array([oovarsIndDict[oov][0] for oov in gradient.keys()])
    Lx2, Ux2 = Lx[:, ind], Ux[:, ind]
    
    A = np.vstack([np.asarray(elem).reshape(1, -1) for elem in gradient.values()]).T
    centers = 0.5 * (Lx2 + Ux2)
    b = np.sum(A * centers, 1) - centerValues.view(np.ndarray) + threshold

#    ind_positive = where(A > 0)
#    ind_negative = where(A < 0)
    
    A_positive = where(A>0, A, 0)
    A_negative = where(A<0, A, 0)
    #S1 = A[ind_positive] * Lx2[ind_positive]
    #S2 = A[ind_negative] * Ux2[ind_negative]
    S1 = A_positive * Lx2
    S2 = A_negative * Ux2
    s1, s2 = np.sum(S1, 1), np.sum(S2, 1)
    S = s1 + s2
    
    alt_threshold1 = where(A_positive != 0, (b.reshape(-1, 1) - S.reshape(-1, 1) + S1) / A_positive, np.inf)
    ind1 = logical_and(Ux2 > alt_threshold1, A_positive != 0)
    Ux2[ind1] = alt_threshold1[ind1]
    
    alt_threshold2 = where(A_negative != 0, (b.reshape(-1, 1) - S.reshape(-1, 1) + S2) / A_negative, -np.inf)
    ind2 = logical_and(Lx2 < alt_threshold2, A_negative != 0)
    Lx2[ind2] = alt_threshold2[ind2]
    
    
    # TODO: check it
    Lx[:, ind], Ux[:, ind] = Lx2, Ux2
    
    # TODO: check indT
    indT[np.any(ind1, 1)] = True
    indT[np.any(ind2, 1)] = True
    
#    for _i, i in enumerate(ind_positive):
#        s = S - S1[:, _i]
#        alt_ub = (b - s) / A[i]
#        ind = Ux[:, i] > alt_ub
#        Ux[ind, i] = alt_ub[ind]
#        indT[ind] = True
#    
#    for _i, i in enumerate(ind_negative):
#        s = S - S2[:, _i]
#        alt_lb = (b - s) / A[i]
#        ind = Lx[:, i] < alt_lb
#        Lx[ind, i] = alt_lb[ind]
#        indT[ind] = True

    ind = np.all(Ux>=Lx, 1)
    if not np.all(ind):
        ind_remain = where(ind)[0]
        lj = ind_remain.size
        Lx = take(Lx, ind_remain, axis=0, out=Lx[:lj])
        Ux = take(Ux, ind_remain, axis=0, out=Ux[:lj])
        indT = indT[ind_remain]
    
    return Lx, Ux, indT, ind_remain
    
'''
def  truncateByConvexFunc():
    if 0 and (p._linear_objective or p.convex in (1, True)) and threshold_prev < 1e300:# and p.convex is True:
    # TODO: rework it
    #centers = dict([(key, val.view(multiarray)) for key, val in centers.items()])
    #centers = dict([(key, 0) for key, val in p._x0.items()])
    

        # TODO: handle indT corrently
        indT2 = np.empty(Lx.shape[0])
        indT2.fill(False)

        #Lx, Ux, indT2 = truncateByPlane(Lx, Ux, indT2, np.hstack([d[v][0] for v in ooVars]), threshold_prev)
        #Lx, Ux, indT2, ind_t = truncateByPlane(Lx, Ux, indT2, np.hstack([d[oov] for oov in ooVars]), threshold_prev)
        
    #        print('==')
    #        print(Lx.sum())
        if p._linear_objective:
            d = p._linear_objective_factor
            th = threshold_prev + (p._linear_objective_scalar if p.goal not in ('min', 'minimum') else - p._linear_objective_scalar)
            Lx, Ux, indT2, ind_t = truncateByPlane(Lx, Ux, indT2, d if p.goal in ('min', 'minimum') else -d, th)
        elif p.convex in (1, True):

            assert p.goal in ('min', 'minimum') 
            wCenters = (Lx+Ux) / 2
            adjustCentersWithDiscreteVariables(wCenters, p)
            #centers = dict([(oovar, asarray((Lx[:, i]+Ux[:, i])/2, dataType)) for i, oovar in enumerate(ooVars)])
            centers = dict([(oovar, asarray(wCenters[:, i], dataType).view(multiarray)) for i, oovar in enumerate(ooVars)])            
            #centers = dict([(key, val.view(multiarray)) for key, val in centers.items()])
            
            #TODO: add other args
            centerValues = fd_obj(centers)
            gradient = fd_obj.D(centers)
            Lx, Ux, indT2, ind_t = truncateByPlane2(centers, centerValues, Lx, Ux, indT2, gradient, threshold_prev, p)
        else:
            assert 0, 'bug in FD kernel'
            
        if ind_t is not True:
            lj = ind_t.size
            Lf = take(Lf, ind_t, axis=0, out=Lf[:lj])
            Uf = take(Uf, ind_t, axis=0, out=Uf[:lj])
            if nlhc is not None:
                nlhc = take(nlhc, ind_t, axis=0, out=nlhc[:lj])
                indTC = np.logical_or(indTC[ind_t], indT2)
            semiinvariant = semiinvariant[ind_t]
            #residual = residual[ind_t]
        
'''





