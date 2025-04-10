#from numpy import isfinite, all, argmax, where, delete, array, asarray, inf, argmin
#from numpy.linalg import norm, solve, LinAlgError
##from openopt.kernel.nonOptMisc import scipyAbsentMsg, scipyInstalled
#import openopt
#from openopt.kernel.setDefaultIterFuncs import SMALL_DELTA_X,  SMALL_DELTA_F
#
#from openopt.kernel.baseSolver import *
#from openopt.kernel.Point import Point
##from openopt.kernel.ooMisc import economyMult, Len
##from openopt.kernel.setDefaultIterFuncs import *
##from UkrOptMisc import getBestPointAfterTurn
#
#App = 1
#
#class lv(baseSolver):
#    __name__ = 'lv'
#    __license__ = "BSD"
#    __authors__ = "Dmitrey"
#    __alg__ = ""
#    __optionalDataThatCanBeHandled__ = ['lb', 'ub']
#    iterfcnConnected = True
#    fStart = None
#    __isIterPointAlwaysFeasible__ = lambda self, p: p.__isNoMoreThanBoxBounded__()
#    #_canHandleScipySparse = True
#
#    #lv default parameters
#
#
#    def __init__(self): pass
#    def __solver__(self, p):
#        if not p.__isFiniteBoxBounded__(): p.err('this solver requires finite lb, ub: lb <= x <= ub')
#        lb, ub = p.lb, p.ub
#        n = p.n
#        f = p.f
#        fTol = p.fTol
#        
#        p.kernelIterFuncs.pop(SMALL_DELTA_X)
#        p.kernelIterFuncs.pop(SMALL_DELTA_F)
#        
#        
#        def getInterval(center, lb, ub, coordToExclude):
#            # center is (lb + ub) / 2.0
#            ooVars = p.freeVarsList
#            domain = dict([(v, (lb[i], ub[i])) for i, v in enumerate(ooVars)])
#            if coordToExclude is not None: 
#                if App == 1:
#                    domain[ooVars[coordToExclude]] = center[coordToExclude]
#                    r = p.user.f[0].interval(domain)
#                    return (r.lb, r.lb)
#                else:
#                    domain[ooVars[coordToExclude]] = (lb[coordToExclude], center[coordToExclude])
#                    r1 = p.user.f[0].interval(domain)
#                    domain[ooVars[coordToExclude]] = (center[coordToExclude], ub[coordToExclude])
#                    r2 = p.user.f[0].interval(domain)
#                    return ((r1.lb, r2.lb), (r1.ub, r2.ub))
#            else:
#                r = p.user.f[0].interval(domain)
#                return r.lb, r.ub
#       
##        def getInterval(center, lb, ub, coordToExclude):
##            # center is (lb + ub) / 2.0
##            ooVars = p.freeVarsList
##            domain = dict([(v, (lb[i], ub[i])) for i, v in enumerate(ooVars)])
##            if coordToExclude is not None: domain[ooVars[coordToExclude]] = center[coordToExclude]
##            r = p.user.f[0].interval(domain)
##            #if coordToExclude is None: raise 0
##            return r.lb, r.ub
#       
#        xRecord = 0.5 * (lb + ub)
#        
#        # doesn't yield correct center, TODO: fix it
##        if p.point(p.x0).betterThan(p.point(xRecord)):
##            xRecord = p.x0
#
#        BestKnownMinValue = p.f(xRecord)    
#        #Centers, Values, fBoxes, xBoxes = [xRecord], [BestKnownMinValue], [getInterval(xRecord)], [(lb, ub)]
#        LBx = [lb]
#        UBx = [ub]
#        fRecord = inf
#        
#        # TODO: maybe rework it
#        fStart = self.fStart
#        if fStart is not None and fStart < BestKnownMinValue: 
#            fRecord = fStart
#        p.extras['nPoints'] = []
#        for itn in range(p.maxIter):
#            #arr = array(Centers, dtype)
#            m = len(UBx)
#            p.extras['nPoints'].append(m)
#            #values = f(asarray(Centers)) #TODO: check does vectorized input work as expected
#            #new_Centers, new_Values, new_fBoxes, new_xBoxes = [], [], [], []
#            #new_xBoxes = [], new_fBoxes = []
#            new_UBx, new_LBx = [], []
#            new_UBf, new_LBf = [], []
#            #print 'm=', m
##            nProc = 2
##            if nProc == 1 or m < 10:
#
#            for i in range(m):
#                #xBox = Centers[i], Values[i], fBoxes[i], xBoxes[i]
#                #lb_x, ub_x = xBox
#                r, r_lb, r_ub = getLbUb(LBx[i], UBx[i], getInterval)
#                new_LBx += r_lb
#                new_UBx += r_ub
#                for (LB, UB) in r:
#                    new_UBf.append(UB)
#                    new_LBf.append(LB)                       
#                    
##            else:
##                from multiprocessing import Pool
##                P = Pool(2)
##                def f2(i):
##                    return getLbUb(LBx[i], UBx[i], getInterval)
##                R = P.map(f2, range(m))
##                raise 0
#
#############################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#            UBx, LBx, UBf, LBf = new_UBx, new_LBx, new_UBf, new_LBf
#            ind = argmin(UBf)
#            xk = (UBx[ind] + LBx[ind]) / 2.0
#            Min = f(xk)
#            p.iterfcn(xk, Min)
#            if p.istop != 0 : return
#            if BestKnownMinValue > Min:
#                BestKnownMinValue = Min
#                xRecord = xk# TODO: is copy required?
#            if fRecord > BestKnownMinValue:
#                fRecord = BestKnownMinValue 
#            
#            th = min((fRecord, BestKnownMinValue - fTol)) 
#            
#            ind = where(asarray(LBf) > th)[0]
#            
#            # TODO: rework it
#            if ind.size != 0: 
#                J = ind.tolist()
#                J.reverse()
#                for j in J:
#                    del UBx[j]
#                    del UBf[j]
#                    del LBx[j]
#                    del LBf[j]
##                    UBx = delete(UBx, ind)
##                    LBx = delete(LBx, ind)
##                    UBf = delete(UBf, ind)
##                    LBf = delete(LBf, ind)
#            if len(UBx) == 0: 
#                p.istop = 1000
#                p.msg = 'optimal solution obtained'
#                break
#        
#        p.iterfcn(xRecord, f(xRecord))
#
#
#
#
#
#def getLbUb(lb_x, ub_x, getInterval):
#    center = 0.5 * (lb_x + ub_x)
#    LB, UB = [], []
#    n = len(lb_x)
#    for j in range(n):
#        lb1_x, ub1_x = lb_x.copy(), ub_x.copy()
#        Lb, Ub = getInterval(center, lb1_x, ub1_x, j)
#        if App == 1:
#            LB.append(Lb)
#            UB.append(Ub)
#        else:
#            LB += Lb
#            UB += Ub
#        
#    Case = 0 # TODO: check other
#    if Case == 0:
#        bestCoordForSplitting = argmin(asarray(UB)-asarray(LB))
#    elif Case == 1:
#        bestCoordForSplitting = argmin(UB)
#    elif Case == 2:
#        bestCoordForSplitting = argmax(LB)
#    if App == 2: bestCoordForSplitting /= 2
#    
##                print '*'*10
##                print asarray(UB)
##                print asarray(LB)
##                print asarray(UB)-asarray(LB)
##                print 'bestCoordForSplitting:', bestCoordForSplitting
##                print '='*10
#    
#    #bestCoordForSplitting = argmax(fBox.ub - fBox.lb)
#    bestCoordForSplittingLength = ub_x[bestCoordForSplitting] - lb_x[bestCoordForSplitting]
#    #print 'bestCoordForSplittingLength:', bestCoordForSplittingLength
#    
#    lb1, ub1 = lb_x.copy(), ub_x.copy()
#    
#    newCoordPosition = (lb1[bestCoordForSplitting] + ub1[bestCoordForSplitting]) / 2.0
#    newLB1 = lb1
#    newLB1[bestCoordForSplitting] = newCoordPosition
#    newUB1 = ub1
#    newUB1[bestCoordForSplitting] = newCoordPosition
#    r, r_lb, r_ub = [], [lb_x, newLB1], [newUB1, ub_x]
#    for L, U in [(lb_x, newUB1), (newLB1, ub_x)]:
#        LB, UB = getInterval(None, L, U, None) # TODO: improve it
#        r.append((LB, UB))
#    return (r, r_lb, r_ub)

from numpy import isfinite, all, argmax, where, delete, array, asarray, inf, argmin
from numpy.linalg import norm, solve, LinAlgError
from openopt.kernel.setDefaultIterFuncs import SMALL_DELTA_X,  SMALL_DELTA_F
from openopt.kernel.baseSolver import *
from openopt.kernel.Point import Point

App = 2

class asdf(baseSolver):
    __name__ = 'asdf'
    __license__ = "BSD"
    __authors__ = "Dmitrey"
    __alg__ = ""
    __optionalDataThatCanBeHandled__ = ['lb', 'ub']
    iterfcnConnected = True
    fStart = None
    __isIterPointAlwaysFeasible__ = lambda self, p: p.__isNoMoreThanBoxBounded__()
    #_canHandleScipySparse = True

    #lv default parameters


    def __init__(self): pass
    def __solver__(self, p):
        if not p.__isFiniteBoxBounded__(): p.err('this solver requires finite lb, ub: lb <= x <= ub')
        p.kernelIterFuncs.pop(SMALL_DELTA_X)
        p.kernelIterFuncs.pop(SMALL_DELTA_F)
        
        lb, ub = p.lb, p.ub
        
        n = p.n
        f = p.f
        fTol = p.fTol
        ooVars = p._freeVarsList
        
        def getInterval(center, lb, ub, coordToExclude):
            # center is (lb + ub) / 2.0
            domain = dict([(v, (lb[i], ub[i])) for i, v in enumerate(ooVars)])
            if coordToExclude is not None: 
                if App == 1:
                    domain[ooVars[coordToExclude]] = center[coordToExclude]
                    r = p.user.f[0].interval(domain)
                    return r.lb, r.ub
                else:
                    domain[ooVars[coordToExclude]] = (lb[coordToExclude], center[coordToExclude])
                    r1 = p.user.f[0].interval(domain)
                    domain[ooVars[coordToExclude]] = (center[coordToExclude], ub[coordToExclude])
                    r2 = p.user.f[0].interval(domain)
                    return ((r1.lb, r2.lb), (r1.ub, r2.ub))
            r = p.user.f[0].interval(domain)
            #if coordToExclude is None: raise 0
            return r.lb, r.ub

#        def getInterval(center, lb, ub, coordToExclude):
#            # center is (lb + ub) / 2.0
#            ooVars = p.freeVarsList
#            domain = dict([(v, (lb[i], ub[i])) for i, v in enumerate(ooVars)])
#            if coordToExclude is not None: domain[ooVars[coordToExclude]] = center[coordToExclude]
#            r = p.user.f[0].interval(domain)
#            #if coordToExclude is None: raise 0
#            return r.lb, r.ub
       
        xRecord = 0.5 * (lb + ub)
        
        # doesn't yield correct center, TODO: fix it
#        if p.point(p.x0).betterThan(p.point(xRecord)):
#            xRecord = p.x0

        BestKnownMinValue = p.f(xRecord)    
        #Centers, Values, fBoxes, xBoxes = [xRecord], [BestKnownMinValue], [getInterval(xRecord)], [(lb, ub)]
        LBx = [lb]
        UBx = [ub]
        fRecord = inf
        
        # TODO: maybe rework it
        fStart = self.fStart
        if fStart is not None and fStart < BestKnownMinValue: 
            fRecord = fStart
        p.extras['nPoints'] = []
        for itn in range(p.maxIter+10):
            #print itn
            m = len(UBx)
            p.extras['nPoints'].append(m)
            new_UBx, new_LBx = [], []
            new_UBf, new_LBf = [], []
            bestCoords = []
            for i in range(m):
                lb_x, ub_x = LBx[i], UBx[i]
                center = 0.5 * (lb_x + ub_x)
                LB, UB = [], []
                for j in range(n):
                    lb1_x, ub1_x = lb_x.copy(), ub_x.copy()
                    Lb, Ub = getInterval(center, lb1_x, ub1_x, j)
                    if App == 1:
                        LB.append(Lb)
                        UB.append(Ub)
                    else:
                        LB += Lb
                        UB += Ub
                        
                Case = 0 # TODO: check other
                if Case == 0:
                    bestCoordForSplitting = argmin(asarray(UB)-asarray(LB))
                elif Case == 1:
                    bestCoordForSplitting = argmin(UB)
                elif Case == 2:
                    bestCoordForSplitting = argmax(LB)
                if App == 2: bestCoordForSplitting /= 2
                
                bestCoords.append(bestCoordForSplitting)
                
                # Debug, remove it
                #bestCoordForSplittingLength = ub_x[bestCoordForSplitting] - lb_x[bestCoordForSplitting]
                #print '!', bestCoordForSplitting, bestCoordForSplittingLength
                
                lb1, ub1 = lb_x.copy(), ub_x.copy()
                
                newCoordPosition = (lb1[bestCoordForSplitting] + ub1[bestCoordForSplitting]) / 2.0
                newLB1 = lb1
                #print 'lb prev:', newLB1[bestCoordForSplitting], ' new:', newCoordPosition
                newLB1[bestCoordForSplitting] = newCoordPosition
                newUB1 = ub1
                #print 'ub prev:', newUB1[bestCoordForSplitting], ' new:', newCoordPosition
                newUB1[bestCoordForSplitting] = newCoordPosition
                new_LBx += [lb_x, newLB1]
                new_UBx += [newUB1, ub_x]
                for L, U in [(lb_x, newUB1), (newLB1, ub_x)]:
                    LB_, UB_ = getInterval(None, L, U, None) # TODO: improve it
                    new_UBf.append(UB_)
                    new_LBf.append(LB_)                       

            if itn > 68: raise 0
            from numpy.linalg import norm
#            print '------'
#            A = [norm(new_UBx[i][1:]-new_LBx[i][1:], 1) for i in range(len(new_UBx))]
#            print sum(A), A
#            B = [norm(UBx[i][1:] - LBx[i][1:], 1) for i in range(len(UBx))]
#            print sum(B), B
#            print '======'
            #UBx, LBx, UBf, LBf = new_UBx, new_LBx, new_UBf, new_LBf
            ind = argmin(new_UBf)
            #print 'LBf1:', LBf
            xk = (new_UBx[ind] + new_LBx[ind]) / 2.0
            Min = f(xk)
#            if Min > p.fk:
#                raise 0

            p.iterfcn(xk, Min)
            if p.istop != 0 : return
            if BestKnownMinValue > Min:
                BestKnownMinValue = Min
                xRecord = xk# TODO: is copy required?
            if fRecord > BestKnownMinValue:
                fRecord = BestKnownMinValue 
            
            th = min((fRecord, BestKnownMinValue - fTol)) 
            
            ind = where(asarray(new_LBf) > th)[0]
            #print 'th:',th, asarray(LBf) - th
            #print ind
            #if itn > 67: raise 0
            
            # TODO: rework it
            if ind.size != 0: 
                J = ind.tolist()
                J.reverse()
                for j in J:
                    del new_UBx[j], new_UBf[j], new_LBx[j], new_LBf[j]
#                    UBx = delete(UBx, ind)
#                    LBx = delete(LBx, ind)
#                    UBf = delete(UBf, ind)
#                    LBf = delete(LBf, ind)
            if len(new_UBx) == 0: 
                p.istop = 1000
                p.msg = 'optimal solution obtained'
                break
            UBx, LBx, UBf, LBf = new_UBx, new_LBx, new_UBf, new_LBf
        
        p.iterfcn(xRecord, f(xRecord))
