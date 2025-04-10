from numpy import hstack,  asarray, abs, atleast_1d, \
logical_not, argsort, vstack, sum, array, nan, all
import numpy as np
from FuncDesigner import oopoint, FDmisc
where = FDmisc.where
#from FuncDesigner.boundsurf import boundsurf


def interalg_ODE_routine(p, solver):
    isIP = p.probType == 'IP'
    isODE = p.probType == 'ODE'
    if isODE:
        f, y0, requiredTimes, ftol = p.equations, p.x0, p.times, p.ftol
        assert len(f) == 1, 'multiple ODE equations are unimplemented for FuncDesigner yet'
        f = list(f.values())[0]
        t = list(f._getDep())[0]
    elif isIP:
        assert p.n == 1 and p.__isNoMoreThanBoxBounded__()
        f, y0, ftol = p.user.f[0], 0.0, p.ftol
        if p.fTol is not None: ftol = p.fTol
        t = list(f._getDep())[0]
        requiredTimes = p.domain[t]
        p.iterfcn(p.point([nan]*p.n))
    else:
        p.err('incorrect prob type for interalg ODE routine')

    eq_var = list(p._x0.keys())[0]

    dataType = solver.dataType
    if type(ftol) == int:
        ftol = float(ftol) # e.g. someone set ftol = 1
    # Currently ftol is scalar, in future it can be array of same length as timeArray
    if len(requiredTimes) < 2:
        p.err('length ot time array must be at least 2')
#    if any(requiredTimes[1:] < requiredTimes[:-1]):
#        p.err('currently interalg can handle only time arrays sorted is ascending order')
#    if any(requiredTimes < 0):
#        p.err('currently interalg can handle only time arrays with positive values')
#    if p.times[0] != 0:
#        p.err('currently solver interalg requires times start from zero')

    deltaT = abs(requiredTimes[-1] - requiredTimes[0])
#    if len(requiredTimes) == 2:
#        requiredTimes = np.linspace(requiredTimes[0], requiredTimes[-1], 150)
    TimesStart = asarray(atleast_1d(requiredTimes[:-1]), dataType)
    TimesEnd = asarray(atleast_1d(requiredTimes[1:]), dataType)

    Diff_store = array([], float)
    TS_store = array([], float)
    TE_store = array([], float)
    maxActiveNodes = 150000#solver.maxActiveNodes

    storedTimesStart = []
    storedTimesEnd = []
    storedValuesSupremum = []
    storedValuesInfinum = []
    currentAllowedResidual = ftol
    F = 0.0
    p._Residual = 0

    # Main cycle
    for itn in range(p.maxIter+1):
        mp = oopoint(
                     {t: [TimesStart, TimesEnd] if requiredTimes[-1] > requiredTimes[0] else [TimesEnd, TimesStart]},
                     skipArrayCast = True
                     )
        mp.isMultiPoint = True
        mp.nPoints = TimesStart.size

        mp.dictOfFixedFuncs = p.dictOfFixedFuncs
        mp._dictOfRedirectedFuncs = p._dictOfRedirectedFuncs
        mp.maxDistributionSize = p.maxDistributionSize
        mp.surf_preference = True
        tmp = f.interval(mp, ia_surf_level = 2 if isIP else 1)
        if not all(tmp.definiteRange):
            p.err('''
            solving ODE and IP by interalg is implemented for definite (real) range only,
            no NaN values in integrand are allowed''')
        # TODO: perform check on NaNs
        isBoundsurf = hasattr(tmp, 'resolve')
        if isBoundsurf:
            if isIP:
                if tmp.level == 1:
                    #adjustCentersWithDiscreteVariables(wCenters, p)
                    centers = oopoint((v, asarray(0.5*(val[0] + val[1]), dataType)) for v, val in mp.items())
                    centers.dictOfFixedFuncs = p.dictOfFixedFuncs
                    centers._dictOfRedirectedFuncs = p._dictOfRedirectedFuncs
                    centers_L, centers_U = tmp.values(centers)
                    Lf, Uf = atleast_1d(centers_L), atleast_1d(centers_U)
                    Diff = Uf-Lf
                    approx_value = 0.5*(Uf+Lf)
                else:
                    assert tmp.level == 2
                    ts, te = TimesStart, TimesEnd
                    A, B = (te**2 + te*ts+ts**2) / 3.0, 0.5 * (te + ts)
                    a, b, c = tmp.l.d2.get(t, 0.0), tmp.l.d.get(t, 0.0), tmp.l.c
                    val_l = a * A + b * B + c
                    a, b, c = tmp.u.d2.get(t, 0.0), tmp.u.d.get(t, 0.0), tmp.u.c
                    val_u =  a * A + b * B + c
                    Diff = val_u - val_l
                    approx_value = 0.5 * (val_l + val_u)
#                    import pylab, numpy
#                    xx = numpy.linspace(-1, 0, 1000)
#                    pylab.plot(xx, tmp.l.d2.get(t, 0.0)[1]*xx**2+ tmp.l.d.get(t, 0.0)[1]*xx+ tmp.l.c[1], 'r')
#                    pylab.plot(xx, tmp.u.d2.get(t, 0.0)[1]*xx**2+ tmp.u.d.get(t, 0.0)[1]*xx+ tmp.u.c[1], 'b')
#                    pylab.grid()
#                    pylab.show()

            elif isODE:
                l, u = tmp.l, tmp.u
                assert len(l.d) <= 1 and len(u.d) <= 1 # at most time variable
#                l_koeffs, u_koeffs = l.d.get(t, 0.0), u.d.get(t, 0.0)
#                l_c, u_c = l.c, u.c
#                dT = TimesEnd - TimesStart if requiredTimes[-1] > requiredTimes[0] else TimesStart - TimesEnd

                ends = oopoint([(v, asarray(val[1], dataType)) for v, val in mp.items()])
                ends.dictOfFixedFuncs = p.dictOfFixedFuncs
                ends._dictOfRedirectedFuncs = p._dictOfRedirectedFuncs
                ends_L, ends_U = tmp.values(ends)

                starts = oopoint((v, asarray(val[0], dataType)) for v, val in mp.items())
                starts.dictOfFixedFuncs = p.dictOfFixedFuncs
                starts._dictOfRedirectedFuncs = p._dictOfRedirectedFuncs
                starts_L, starts_U = tmp.values(starts)

#                Lf, Uf = atleast_1d(centers_L), atleast_1d(centers_U)

                Lf, Uf = tmp.resolve()[0]
#                Diff = 0.5 * u_koeffs * dT  + u_c  - (0.5 * l_koeffs * dT  + l_c)
                Diff_end = 0.5 * (ends_U - ends_L)
                Diff_start = 0.5 * (starts_U - starts_L)
                Diff = where(Diff_end>Diff_start, Diff_end, Diff_start)

#                Diff = 0.5 * u_koeffs * dT ** 2 + u_c * dT - (0.5 * l_koeffs * dT ** 2 + l_c * dT)
#                Diff =  0.5*u_koeffs * dT  + u_c  - ( 0.5*l_koeffs * dT  + l_c)

#                Lf = 0.5*l_koeffs * dT + l_c
#                Uf = 0.5*u_koeffs * dT + u_c
                #assert 0, 'unimplemented'
            else:
                assert 0
        else:
            Lf, Uf = atleast_1d(tmp.lb), atleast_1d(tmp.ub)
            ends_L = starts_L = Lf
            ends_U = starts_U = Uf
            Diff = Uf - Lf
            approx_value = 0.5 * (Uf+Lf)

        if isODE:
            ind_s = atleast_1d(Diff <= 0.95 * currentAllowedResidual)
            ind_s = np.logical_and(ind_s, Diff < ftol)
            ind_s = np.logical_and(ind_s, Uf-Lf < ftol)
#        else:
#            ind_s = atleast_1d(Diff <= 0.95 * currentAllowedResidual / deltaT)


        if isODE and isBoundsurf:
            d = deltaT #if not isODE or not isBoundsurf else len(TimesStart)
            ind_s = np.logical_and(
                                atleast_1d(Diff_end <= 0.95 * currentAllowedResidual / d),
                                atleast_1d(Diff_start <= 0.95 * currentAllowedResidual / d)
                                )
            ind_s &= atleast_1d(Diff_end <= ftol)
            ind_s &= atleast_1d(Diff_start <= ftol)
        else:
            ind_s = atleast_1d(Diff <= 0.95 * currentAllowedResidual / deltaT)

#            ind_s = np.logical_and(ind_s, Diff < ftol)
#            ind_s = np.logical_and(ind_s, Uf-Lf < ftol)

        ind = where(ind_s)[0]
        if isODE:
            storedTimesStart.append(TimesStart[ind])
            storedTimesEnd.append(TimesEnd[ind])
            storedValuesSupremum.append(Uf[ind])
            storedValuesInfinum.append(Lf[ind])
#            storedValuesSupremum.append(ends_U[ind])
#            storedValuesInfinum.append(ends_L[ind])
        else:
            assert isIP
            #F += 0.5 * sum((TimesEnd[ind]-TimesStart[ind])*(Uf[ind]+Lf[ind]))
            F += sum((TimesEnd[ind]-TimesStart[ind])*approx_value[ind])

        if ind.size != 0:
            tmp = abs(TimesEnd[ind] - TimesStart[ind])
            Tmp = sum(Diff[ind] * tmp) #if not isODE or not isBoundsurf else sum(Diff[ind])
            currentAllowedResidual -= Tmp
            if isIP: p._residual += Tmp
            deltaT -= sum(tmp)

        ind = where(logical_not(ind_s))[0]
        if 1:#new
            if ind.size == 0 and Diff_store.size == 0:
                p.istop = 1000
                p.msg = 'problem has been solved according to required user-defined accuracy %0.1g' % ftol
                break
            if ind.size != 0:
                # TODO: use merge sorted lists
                if Diff_store.size != 0:
                    Diff_store = hstack((Diff_store, Diff[ind]*abs(TimesEnd[ind] - TimesStart[ind])))
                    TS_store =  hstack((TS_store, TimesStart[ind]))
                    TE_store =  hstack((TE_store, TimesEnd[ind]))
                else:
                    Diff_store = Diff[ind]*abs(TimesEnd[ind] - TimesStart[ind])
                    TS_store, TE_store = TimesStart[ind], TimesEnd[ind]
                ind_a = argsort(Diff_store)
                Diff_store = Diff_store[ind_a]
                TS_store = TS_store[ind_a]
                TE_store = TE_store[ind_a]
            p.extras['nNodes'].append(TS_store.size)

            TS_store, TS = TS_store[:-maxActiveNodes], TS_store[-maxActiveNodes:]
            TE_store, TE = TE_store[:-maxActiveNodes], TE_store[-maxActiveNodes:]
            Diff_store = Diff_store[:-maxActiveNodes]
            middleTimes = 0.5 * (TS + TE)
            p.extras['nActiveNodes'].append(middleTimes.size)
            TimesStart = vstack((TS, middleTimes)).flatten()
            TimesEnd = vstack((middleTimes, TE)).flatten()
        else:
            if ind.size == 0:
                p.istop = 1000
                p.msg = 'problem has been solved according to required user-defined accuracy %0.1g' % ftol
                break

            TS, TE = TimesStart[ind], TimesEnd[ind]
            middleTimes = 0.5 * (TS + TE)
            TimesStart = vstack((TS, middleTimes)).flatten()
            TimesEnd = vstack((middleTimes, TE)).flatten()

        # !!! unestablished !!!
        if isODE:
            p.iterfcn(fk = currentAllowedResidual/ftol)
        elif isIP:
            p.iterfcn(xk=array(nan), fk=F, rk = ftol - currentAllowedResidual)
        else:
            p.err('bug in interalgODE.py')

        if p.istop != 0 :
            break

        #print(itn, TimesStart.size)

    if isODE:

        t0, t1, lb, ub = hstack(storedTimesStart), hstack(storedTimesEnd), hstack(storedValuesInfinum), hstack(storedValuesSupremum)
        ind = argsort(t0)
        if requiredTimes[0] > requiredTimes[-1]:
            ind = ind[::-1] # reverse
        t0, t1, lb, ub = t0[ind], t1[ind], lb[ind], ub[ind]
        lb, ub = hstack((y0, y0+(lb*(t1-t0)).cumsum())), hstack((y0, y0+(ub*(t1-t0)).cumsum()))
        #y_var = p._x0.keys()[0]
        #p.xf = p.xk = 0.5*(lb+ub)
        p.extras = {'startTimes': t0, 'endTimes': t1, eq_var:{'infinums': lb, 'supremums': ub}}
        return t0, t1, lb, ub
    elif isIP:
        P = p.point([nan]*p.n)
        P._f = F
        P._mr = ftol - currentAllowedResidual
        P._mrName = 'None'
        P._mrInd = 0
#        p.xk = array([nan]*p.n)
#        p.rk = currentAllowedResidual
#        p.fk = F
        #p._Residual =
        p.iterfcn(asarray([nan]*p.n), fk=F, rk = ftol - currentAllowedResidual)
    else:
        p.err('incorrect prob type in interalg ODE routine')
