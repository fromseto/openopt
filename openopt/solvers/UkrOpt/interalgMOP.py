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

def r43_seq(Arg):
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
            o, a = lf[:, i], uf[:, i] 
            if val == inf:
                ff = s[i] + tol
                ind = a > ff
                if any(ind):
                    t1 = a[ind]
                    t2 = o[ind]
                    t_diff = t1-t2
                    t_diff[t_diff<1e-200] = 1e-200
                    # TODO: check discrete cases
                    Tmp = (ff-t2) / t_diff

                    tmp[ind] *= Tmp
                    tmp[ff<o] = 0.0
                    
            elif val == -inf:
                ff = s[i] - tol
                ind = o < ff
                if any(ind):
                    t1 = a[ind]
                    t2 = o[ind]
                    t_diff = t1-t2
                    t_diff[t_diff<1e-200] = 1e-200
                    
                    Tmp = (t1-ff) / t_diff

                    tmp[ind] *= Tmp
                    tmp[a<ff] = 0.0
            else: # finite val
                ff = abs(s[i]-val) - tol
                if ff <= 0:
                    continue
                _lf, _uf = o - val, a - val
                ind = logical_or(_lf < ff, _uf > - ff)
                _lf = _lf[ind]
                _uf = _uf[ind]
                _lf[_lf>ff] = ff
                _lf[_lf<-ff] = -ff
                _uf[_uf<-ff] = -ff
                _uf[_uf>ff] = ff
                
                r20 = a[ind] - o[ind]
                r20[r20<1e-200] = 1e-200
                _diff = _uf - _lf
                _diff[_diff<1e-200] = 1e-200
                
                Tmp = 1.0 - (_uf - _lf) / r20
                
                tmp[ind] *= Tmp

        new = 0
        if new:
            ind_0 = tmp == 0.0
            ind_1 = tmp == 1.0
            r[ind_1] = inf
            ind_m = logical_not(logical_and(ind_0, ind_1))
            #r[ind_m] -= log1p(-tmp[ind_m]) * 1.4426950408889634
            r[ind_m] -= log1p(-tmp[ind_m]) * 1.4426950408889634
            #r20 = log2(1-tmp[ind_m]) #* 1.4426950408889634
            
        else:
            r -= log1p(-tmp) * 1.4426950408889634 # log2(e)
            
        #r -= r20
    return r

from multiprocessing import Pool
def r43(targets, SolutionsF, lf, uf, pool, nProc):
    lf, uf = asarray(lf), asarray(uf)
    target_vals = [t.val for t in targets]
    target_tols = [t.tol for t in targets]
    if nProc == 1 or len(SolutionsF) <= 1:
        return r43_seq((target_vals, target_tols, SolutionsF, lf, uf))
        
    splitBySolutions = True #if len(SolutionsF) > max((4*nProc, ))
    if splitBySolutions:
        ss = array_split(SolutionsF, nProc)
        Args = [(target_vals, target_tols, s, lf, uf) for s in ss]
        result = pool.imap_unordered(r43_seq, Args)#, callback = cb)    
        r = [elem for elem in result if elem is not None]
        return PythonSum(r)
    else:
        lf2 = array_split(lf, nProc)
        uf2 = array_split(uf, nProc)
        Args = [(target_vals, target_tols, SolutionsF, lf2[i], uf2[i]) for i in range(nProc)]
        result = pool.map(r43_seq, Args)
        r = [elem for elem in result if elem is not None]
        return vstack(r)

def r14MOP(p, nlhc, residual, definiteRange, y, e, vv, asdf1, C, r40, g, nNodes,  \
         r41, fTol, Solutions, varTols, _in, dataType, \
         maxNodes, _s, indTC, xRecord, activeCons):

    assert p.probType == 'MOP'
    
    asdf1 = [t.func for t in p.targets]
    
#    if len(p._discreteVarsNumList):
#        y, e, ind_trunc = adjustDiscreteVarBounds(y, e, p)
#        if ind_trunc.size:
#            
#            _s, indTC, 
    
    
    if p.nProc != 1 and getattr(p, 'pool', None) is None:
        p.pool = Pool(processes = p.nProc)
    elif p.nProc == 1:
        p.pool = None
    
    ol, al = [], []
    targets = p.targets # TODO: check it
    m, n = y.shape
    ol, al = [[] for k in range(m)], [[] for k in range(m)]
    ind_nan = np.zeros(m, bool)
    for i, t in enumerate(targets):
        y, e, o, a, definiteRange, exactRange, _s_, indTC_ = func82(y, e, vv, t.func, dataType, p)
#        o, a, definiteRange, exactRange, _s, indTC = func82(y, e, vv, t.func, dataType, p, _s, indTC)
        o, a = o.reshape(2*n, m).T, a.reshape(2*n, m).T
        ind_nan |= logical_and(all(isnan(o), axis=1), np.all(isnan(a), axis=1))
        for j in range(m):
            ol[j].append(o[j])
            al[j].append(a[j])
        #ol.append(o.reshape(2*n, m).T.tolist())
        #al.append(a.reshape(2*n, m).T.tolist())

    fo_prev = 0
    
    if y.size == 0:
        return _in, g, fo_prev, _s, Solutions, xRecord, r41, r40
    
    nodes, y, e, o, a, _s, indT, nlhc, residual, activeCons = func11(y, e, nlhc, indTC, residual, ol, al, _s, p, activeCons, ind_nan)
    if len(nodes) == 0: # after remove some nodes with NaNs
        return _in, g, fo_prev, _s, Solutions, xRecord, r41, r40
        
    m, n = y.shape # updated value
    assert _s.size == m == len(ol) == y.shape[0]
    
    nlh_obj = r43(targets, Solutions.F, ol, al, p.pool, p.nProc)
    
    #y, e = func4(y, e, o, a, fo)
    
    
    assert p.solver.dataHandling == 'raw', '"sorted" mode is unimplemented for MOP yet'
    
    if nlh_obj is None:
        new_nodes_tnlh_all = nlhc
    elif nlhc is None: 
        new_nodes_tnlh_all = nlh_obj
    else:
        new_nodes_tnlh_all = nlh_obj + nlhc

    r5F, r5Coords = getr4Values(vv, y, e, activeCons, new_nodes_tnlh_all, asdf1, C, p.contol, dataType, p) 

    fo = 0 # unused for MOP
    
    nIncome, nOutcome = r44(Solutions, r5Coords, r5F, targets, p.solver.sigma)
    
    p._frontLength = len(Solutions.F)
    p._nIncome = nIncome
    p._nOutcome = nOutcome
    p.iterfcn(p.x0)
    #print('iter: %d (%d) frontLenght: %d' %(p.iter, itn, len(Solutions.coords)))


    # TODO: better of nlhc for unconstrained probs

#    if len(_in) != 0:
#        an = hstack((nodes,  _in))
#    else:
#        an = atleast_1d(nodes)

    an = nodes + _in

    
    if p.istop != 0: 
        return an, g, fo, None, Solutions, xRecord, r41, r40
        
    

    

    hasNewParetoNodes = False if nIncome == 0 else True
    if hasNewParetoNodes:
        nodesForRecalculation = nodes + _in
        ol2 = [node.o for node in nodesForRecalculation]
        al2 = [node.a for node in nodesForRecalculation]
        nlhc2 = [node.nlhc for node in nodesForRecalculation]
        nlh_obj2 = r43(targets, Solutions.F, ol2, al2, p.pool, p.nProc)
        tnlh_all = asarray(nlhc2) if nlh_obj2 is None else nlh_obj2 if nlhc2[0] is None else asarray(nlhc2) + nlh_obj2
        
        r10 = logical_not(any(isfinite(tnlh_all), 1))
        if any(r10):
            ind = where(logical_not(r10))[0]
            an = [an[i] for i in ind]
            tnlh_all = take(tnlh_all, ind, axis=0, out=tnlh_all[:ind.size])        
        work_elems = (an, tnlh_all)
    else:
        work_elems = (nodes, new_nodes_tnlh_all)
    
    Nodes, Tnlh_all = work_elems
    r10 = logical_not(any(isfinite(Tnlh_all), 1))
    if any(r10):
        ind = where(logical_not(r10))[0]
        for i in where(r10)[0][::-1]:
            Nodes.pop(i)
        if hasNewParetoNodes:
            Tnlh_all = tnlh_all = take(tnlh_all, ind, axis=0, out=tnlh_all[:ind.size])
            an = nodes + _in 
        else:
            Tnlh_all = new_nodes_tnlh_all = take(new_nodes_tnlh_all, ind, axis=0, out=new_nodes_tnlh_all[:ind.size])
    
    if len(an) == 0:
        return _in, g, fo_prev, _s, Solutions, xRecord, r41, r40
        
    

    n = Tnlh_all.shape[1] // 2
    T1, T2 = Tnlh_all[:, :n], Tnlh_all[:, n:]
    T = where(logical_or(T1 < T2, isnan(T2)), T1, T2)
    t = asarray(nanargmin(T, 1), int)
    
    w = arange(t.size)
    NN = T[w, t].flatten()# TODO: IS flatten() required here?

    for i, node in enumerate(Nodes):
        # low priority, mb rework/remove it
        node.tnlh_curr_best = NN[i]# used only for estimation of number of nodes to involve
        
    if p.solver.mop_mode == 1:
        for i, node in enumerate(Nodes):
            node.tnlh_all = Tnlh_all[i] # TODO: rework/remove it
    elif len(nodes) != 0:
        yc, ec = vstack([n.y for n in nodes]), vstack([n.e for n in nodes])
        oc, ac = vstack([n.o for n in nodes]), vstack([n.a for n in nodes])
        
        # !!!!!!! TODO: implement o, a truncution for MOP there !!!!!!
        indT = func4(p, yc, ec, oc, ac, fo, new_nodes_tnlh_all)
        
        _indT = array([n.indtc for n in nodes])
        indT &= _indT
        for i, n in enumerate(nodes):
            n.indtc = indT[i]
        
        

    astnlh = argsort(NN)

    if hasNewParetoNodes:
        an = [an[i] for i in astnlh]
        p._t = t
    else:# Nodes is nodes
        nodes = [nodes[i] for i in astnlh]
        an = nodes + _in
        tmp = getattr(p, '_t', [])
       
        p._t = np.hstack((t, tmp)) if len(tmp) else t

    
    
    #assert _s.size == m == len(ol) == y.shape[0]
    
    
    # TODO: form _s in other level (for active nodes only), to reduce calculations
#    if len(an) != 0:
#        T = asarray([node.nlh_obj_fixed for node in an])
##        nlhc_fixed = asarray([node.nlhc for node in an])
#        if an[0].nlhc is not None:
#            T += asarray([node.nlhc for node in an])
##        T = nlhf_fixed + nlhc_fixed if nlhc_fixed[0] is not None else nlhf_fixed 
#        p.__s = \
#        nanmin(vstack(([T[w, t], T[w, n+t]])), 0)
#    else:
#        p.__s = array([])

#        p._nObtainedSolutions = len(solutions)
#        if p._nObtainedSolutions > maxSolutions:
#            solutions = solutions[:maxSolutions]
#            p.istop = 0
#            p.msg = 'user-defined maximal number of solutions (p.maxSolutions = %d) has been exeeded' % p.maxSolutions
#            return an, g, fo, None, solutions, coords, xRecord, r41, r40
    
    

    #an, g = func9(an, fo, g, p)

    nn = maxNodes#1 if asdf1.isUncycled and all(isfinite(o)) and p._isOnlyBoxBounded and not p.probType.startswith('MI') else maxNodes
    
    an, g = func5(an, nn, g, p)
    nNodes.append(len(an))
    
    assert _s.size == m == len(ol) == y.shape[0]
    
    return an, g, fo, _s, Solutions, xRecord, r41, r40



def r44(Solutions, r5Coords, r5F, targets, sigma):
#    print Solutions.F
#    if len(Solutions.F) != Solutions.coords.shape[0]:
#        raise 0
    # TODO: rework it
    #sf = asarray(Solutions.F)
    nIncome, nOutcome = 0, 0
    m= len(r5Coords)
    #n = len(r5Coords[0])
    # TODO: mb use inplace r5Coords / r5F modification instead?
    for j in range(m):
        if np.any(isnan(r5F[j])):
            continue
        if Solutions.coords.size == 0:
            Solutions.coords = array(r5Coords[j]).reshape(1, -1)
            Solutions.F.append(r5F[0])
            nIncome += 1
            continue
        M = Solutions.coords.shape[0] 
        
        r47 = empty(M, bool)
        r47.fill(False)
#        r48 = empty(M, bool)
#        r48.fill(False)
        for i, target in enumerate(targets):
            
            f = r5F[j][i]
            
            # TODO: rewrite it
            F = asarray([Solutions.F[k][i] for k in range(M)])
            #d = f - F # vector-matrix
            
            val, tol = target.val, target.tol
            Tol = sigma * tol
            if val == inf:
                r52 = f > F + Tol
#                r36olution_better = f <= F#-tol
            elif val == -inf:
                r52 = f < F - Tol
#                r36olution_better = f >= F#tol
            else:
                r52 = abs(f - val) < abs(F - val) - Tol
#                r36olution_better = abs(f - val) >= abs(Solutions.F[i] - val)#-tol # abs(Solutions.F[i] - target)  < abs(f[i] - target) + tol
            
            r47 = logical_or(r47, r52)
#            r48 = logical_or(r48, r36olution_better)
        
        accept_c = all(r47)
        #print sum(asarray(Solutions.F))/asarray(Solutions.F).size
        if accept_c:
            nIncome += 1
            #new
            r48 = empty(M, bool)
            r48.fill(False)
            for i, target in enumerate(targets):
                f = r5F[j][i]
                F = asarray([Solutions.F[k][i] for k in range(M)])
                val, tol = target.val, target.tol
                if val == inf:
                    r36olution_better = f < F
                elif val == -inf:
                    r36olution_better = f > F
                else:
                    r36olution_better = abs(f - val) > abs(F - val)
                r48 = logical_or(r48, r36olution_better)

            r49 = logical_not(r48)
            remove_s = any(r49)
            if remove_s:
                r50 = where(r49)[0]
                nOutcome += r50.size
                Solutions.coords[r50[0]] = r5Coords[j]
                Solutions.F[r50[0]] = r5F[j]
                
                if r50.size > 1:
                    r49[r50[0]] = False
                    indLeft = logical_not(r49)
                    indLeftPositions = where(indLeft)[0]
                    newSolNumber = Solutions.coords.shape[0] - r50.size + 1
                    Solutions.coords = take(Solutions.coords, indLeftPositions, axis=0, out = Solutions.coords[:newSolNumber])
                    Solutions.F = [Solutions.F[i] for i in indLeftPositions]
            else:
                Solutions.coords = vstack((Solutions.coords, r5Coords[j]))
                Solutions.F.append(r5F[j])
    return nIncome, nOutcome
