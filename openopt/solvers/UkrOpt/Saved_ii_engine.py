from interalgLLR import *
from numpy import inf, prod, all, sum, zeros
#from FuncDesigner.boundsurf import boundsurf

# for PyPy
from openopt.kernel.nonOptMisc import where

def processBoxesIP(p, nlhc, residual, definiteRange, Lx, Ux, ooVars, fd_obj, C, CurrentBestKnownPointsMinValue, cutLevel, nNodes,  \
         fRecord, fTol, Solutions, varTols, inactiveNodes, dataType, \
         maxNodes, semiinvariant, indTC, xRecord, activeCons):

    required_sigma = p.ftol * 0.99 # to suppress roundoff effects

    m, n = Lx.shape

    ip = formIntervalPoint(Lx, Ux, ooVars)
    ip.dictOfFixedFuncs = p.dictOfFixedFuncs
    ip._dictOfRedirectedFuncs = p._dictOfRedirectedFuncs
    ip.maxDistributionSize = p.maxDistributionSize
    ip.surf_preference = True

    tmp = fd_obj.interval(ip, ia_surf_level=2)
#    print(type(tmp))
    if hasattr(tmp, 'resolve'):#type(tmp) == boundsurf:
#            print('b')
        #adjustCentersWithDiscreteVariables(wCenters, p)

        if 1:
        # changes
            val_l, val_u = zeros(2*n*m), zeros(2*n*m)

            val_l += tmp.l.c # may be array <- scalar
            val_u += tmp.u.c

            for v in tmp.dep:
                ind = p._oovarsIndDict[v]
                ts, te = ip[v]#Lx[:, ind], Ux[:, ind]
                A, B = (te**2 + te*ts + ts**2) / 3.0, 0.5 * (te + ts)
                #A, B, C = (te**3 - ts**3) / 3.0, 0.5 * (te**2 - ts**2), te - ts
                b = tmp.l.d.get(v, 0.0)
                val_l += b * B
                if tmp.level == 2:
                    a = tmp.l.d2.get(v, 0.0)
                    val_l += a * A

                b = tmp.u.d.get(v, 0.0)
                val_u +=  b * B
                if tmp.level == 2:
                    a = tmp.u.d2.get(v, 0.0)
                    val_u += a * A

            #Diff = val_u - val_l
            #approx_value = 0.5 * (val_l + val_u)
            Lf, Uf = val_l, val_u
        # changes end
        else:
            centers = oopoint((v, asarray(0.5*(val[0] + val[1]), dataType)) for v, val in ip.items())
            centers.dictOfFixedFuncs = p.dictOfFixedFuncs
            centers._dictOfRedirectedFuncs = p._dictOfRedirectedFuncs
            centers.maxDistributionSize = p.maxDistributionSize

            Lf, Uf = tmp.values(centers)

        definiteRange = tmp.definiteRange
    else:
        Lf, Uf, definiteRange = tmp.lb, tmp.ub, tmp.definiteRange


    if not all(definiteRange):
        p.err('''
        numerical integration with interalg is implemented
        for definite (real) range only, no NaN values in integrand are allowed''')

    Lf, Uf = Lf.reshape(2*n, m).T, Uf.reshape(2*n, m).T

    nodes, Lx, Ux, Lf, Uf, semiinvariant, indT, nlhc, residual, activeCons = formNodes(Lx, Ux, None, indTC, None, Lf, Uf, semiinvariant, p, activeCons)

    #AllNodes = nodes if len(inactiveNodes) == 0 else hstack((inactiveNodes, nodes)).tolist()
    AllNodes = nodes + inactiveNodes

    if 1:
        AllNodes.sort(key = lambda obj: obj.key, reverse=False)
    else:
        AllNodes.sort(key=lambda obj: obj.volumeResidual, reverse=False)

    UfLf_diff = array([node.key for node in AllNodes])
    volumes = array([node.volume for node in AllNodes])

    IND = UfLf_diff <= 0.95*(required_sigma-p._residual) / (prod(p.ub-p.lb) - p._volume)
    ind = where(IND)[0]
    # TODO: use true_sum
    v = volumes[ind]
    p._F += sum(array([AllNodes[i].F for i in ind]) * v)
    residuals = UfLf_diff[ind] * v
    p._residual += residuals.sum()
    p._volume += v.sum()

    #AllNodes = asarray(AllNodes, object)
    #AllNodes = AllNodes[where(logical_not(IND))[0]]
    AllNodes = [elem for i,  elem in enumerate(AllNodes) if not IND[i]]

    nNodes.append(len(AllNodes))

    p.iterfcn(xk=array(nan), fk=p._F, rk = 0)#TODO: change rk to something like p._r0 - p._residual
    if p.istop != 0:
        UfLf_diff = array([node.key for node in AllNodes])
        volumes = array([node.volume for node in AllNodes])
        p._residual += sum(UfLf_diff * volumes)
        semiinvariant = None

    #AllNodes, cutLevel = removeSomeNodes(AllNodes, threshold, cutLevel, 'IP')
    #nCut = 1 if fd_obj.isUncycled and all(isfinite(Uf)) and all(isfinite(Lf)) and p._isOnlyBoxBounded else maxNodes
    #AllNodes, cutLevel = TruncateOutOfAllowedNumberNodes(AllNodes, nCut, cutLevel)

    return AllNodes, cutLevel, inf, semiinvariant, Solutions, xRecord, fRecord, CurrentBestKnownPointsMinValue
