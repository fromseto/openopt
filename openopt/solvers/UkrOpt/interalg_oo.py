PythonMin, PythonMax = min, max
PythonAll = all
import numpy as np

from sortedcontainers import SortedDict

from openopt.kernel.setDefaultIterFuncs import SMALL_DELTA_X,  SMALL_DELTA_F, MAX_NON_SUCCESS, IS_NAN_IN_X
from openopt.kernel.baseSolver import *

from openopt.solvers.UkrOpt.interalgMisc import *
from FuncDesigner import sum as fd_sum, abs as fd_abs, oopoint
from ii_engine import *


#from interalgT import getTruncatedArrays


bottleneck_is_present = False
try:
    from bottleneck import nanmin
    bottleneck_is_present = True
except ImportError:
    from numpy import nanmin



class interalg(baseSolver):
    __name__ = 'interalg'
    __license__ = "BSD"
    __authors__ = "Dmitrey"
    __alg__ = ""
    __optionalDataThatCanBeHandled__ = [\
    'lb', 'ub', 'c', 'h', 'A', 'Aeq', 'b', 'beq', 'discreteVars', 'QC', 'intVars'\
    ]
    iterfcnConnected = True
    fStart = None
    dataType = np.float64
    #maxMem = '150MB'
    maxNodes = 150000
    minActiveNodes = 15
    maxActiveNodes = 150
    sigma = 0.1 # for MOP, unestablished yet
    
    _requiresBestPointDetection = True
    
    __isIterPointAlwaysFeasible__ = lambda self, p: \
    p.__isNoMoreThanBoxBounded__() or p.probType in ('MOP', 'IP') #and p.probType != 'IP'
    _requiresFiniteBoxBounds = True

    _constraintInactiveValue = -1000.0 # ~= log2(1e-300), 
    # a value for _inner_ interalg purposes, not intended for solver users
    
    _constraints_reduction = True
    prioritized_constraints = True

    mop_mode = 2

    def __init__(self): 
        self.dataHandling = 'auto'
        
    def __solver__(self, p):
        if not p.isFDmodel:
            p.err('solver %s can handle only FuncDesigner problems' % self.__name__)
            
        isMOP = p.probType == 'MOP'
        if isMOP:
            from interalgMOP import r14MOP
        #isOpt = p.probType in ['NLP', 'NSP', 'GLP', 'MINLP']
        isODE = p.probType == 'ODE'
        isIP = p.probType == 'IP'
        isSNLE = p.probType in ('NLSP', 'SNLE')
        
        if not isIP and not isODE:
            if 1:
                from interalgCons import processConstraints
                use_saved = 0
            else:
                from Saved_interalgCons import processConstraints
                use_saved = 1

        if not p.__isFiniteBoxBounded__() and not isODE: 
            p.err('''
            solver %s requires finite lb, ub: 
            lb <= x <= ub 
            (you can use "implicitBounds"),
            e.g. p.implicitBounds = (-100, 100)
            or mere p.implicitBounds = 100
            ''' % self.__name__)
#        if p.fixedVars is not None:
#            p.err('solver %s cannot handle FuncDesigner problems with some variables declared as fixed' % self.__name__)
#        if p.probType in ('LP', 'MILP'):
#            p.err("the solver can't handle problems of type " + p.probType)

        for v in p.freeVarsSet:
            if v.domain is int or v.domain is 'int':
                p.err('interalg cannot handle intVars yet, use domain = [val1, val2, ...,val_k]')

        dataType = self.dataType
        if type(dataType) == str:
            if not hasattr(np, dataType):
                p.pWarn('your architecture has no type "%s", float64 will be used instead' % dataType)
                dataType = 'float64'
            dataType = getattr(np, dataType)
            self.dataType = dataType
        
        vv = list(p._freeVarsList)
        
        #TODO: create it in openopt kernel, not in interalg kernel
        p._freeVarsDict = dict((v, i) for i, v in enumerate(vv))
        
        
        x0 = dict((v, p._x0[v]) for v in vv)
        domain = oopoint(((v, [atleast_1d(p.lb[i]), atleast_1d(p.ub[i])]) for i,  v in enumerate(vv)), skipArrayCast = True)
        domain.dictOfFixedFuncs = p.dictOfFixedFuncs
        domain._dictOfRedirectedFuncs = p._dictOfRedirectedFuncs
        domain._dictOfStochVars = p._dictOfStochVars
        domain._p = p
        domain.maxDistributionSize = p.maxDistributionSize
        
#                domain = ooPoint(domainData, skipArrayCast=True)
#        domain.isMultiPoint = True
#        domain.nPoints = m
#        domain.dictOfFixedFuncs = p.dictOfFixedFuncs
#        domain._dictOfRedirectedFuncs = p._dictOfRedirectedFuncs
#        domain.maxDistributionSize = p.maxDistributionSize
#        domain._dictOfStochVars = p._dictOfStochVars
#        domain._p = p
        
        
        if isIP:
            pb = r14IP
            p._F = asarray(0, self.dataType)
            p._residual = 0.0
            f_int = p.user.f[0].interval(domain, self.dataType)
            p._r0 = prod(p.ub-p.lb) * (f_int.ub - f_int.lb)
            p._volume = 0.0
            p.kernelIterFuncs.pop(IS_NAN_IN_X)
        elif isMOP:
            pb = r14MOP
        else:
            pb = r14
            if not isSNLE:
                f_dep = p.user.f[0].dep
                dep = f_dep & p.freeVarsSet
                p._continuous_obj_dep_variables = PythonAll(v.domain is None for v in dep)
                

        
        for val in p._x0.values():
            if isinstance(val,  (list, tuple, np.ndarray)) and len(val) > 1:
                p.pWarn('''
                solver %s currently can handle only single-element variables, 
                use oovars(n) instead of oovar(size=n),
                elseware correct result is not guaranteed
                '''% self.__name__)

        
        for val in x0.values():
            if isinstance(val,  (list, tuple, np.ndarray)) and len(val) > 1:
                p.err('''
                solver %s currently can handle only single-element variables, 
                use oovars(n) instead of oovar(size=n)'''% self.__name__)

        point = p.point
        
        p.kernelIterFuncs.pop(SMALL_DELTA_X, None)
        p.kernelIterFuncs.pop(SMALL_DELTA_F, None)
        p.kernelIterFuncs.pop(MAX_NON_SUCCESS, None)
        
#        if not bottleneck_is_present and not isODE:
#                p.pWarn('''
#                installation of Python module "bottleneck" 
#                (http://berkeleyanalytics.com/bottleneck,
#                available via easy_install, takes several minutes for compilation)
#                could speedup the solver %s''' % self.__name__)
        
        n = p.n
        
        maxSolutions = p.maxSolutions
        if maxSolutions == 0: maxSolutions = 10**50
        if maxSolutions != 1 and p.fEnough != -np.inf:
            p.warn('''
            using the solver interalg with non-single solutions mode 
            is not ajusted with fEnough stop criterium yet, it will be omitted
            ''')
            p.kernelIterFuncs.pop(FVAL_IS_ENOUGH)
        
        nNodes = []        
        p.extras['nNodes'] = nNodes
        nActiveNodes = []
        p.extras['nActiveNodes'] = nActiveNodes

        Solutions = Solution()
        Solutions.maxNum = maxSolutions
        Solutions.solutions = []
        Solutions.coords = np.array([]).reshape(0, n)
        p.solutions = Solutions
        
        lb, ub = asarray(p.lb, dataType).copy(), asarray(p.ub, dataType).copy()

        fTol = p.fTol
        if isIP or isODE:
            if p.ftol is None:
                if fTol is not None:
                    p.ftol = fTol
                else:
                    p.err('interalg requires user-supplied ftol (required precision)')
            if fTol is None: fTol = p.ftol
            elif fTol != p.ftol:
                p.err('you have provided both ftol and fTol')

        if fTol is None and not isMOP: # TODO: require tols for MOP
            fTol = 1e-7
            p.warn('''
            solver %s requires p.fTol value (required objective function tolerance); 
            10^-7 will be used''' % self.__name__
            )

        xRecord = 0.5 * (lb + ub)
        
        # TODO: rework it if discrete variables will be somehow added for IP or ODE
        if not isIP and not isODE:
            from interalgLLR import adjustr4WithDiscreteVariables
            adjustr4WithDiscreteVariables(xRecord.reshape(1, -1), p)

        r40 = np.inf
        
        y = lb.reshape(1, -1)
        e = ub.reshape(1, -1)
        r41 = np.inf

        # TODO: maybe rework it, especially for constrained case
        fStart = self.fStart
        
        # TODO: remove it after proper SNLE handling implementation
        if isSNLE:
            r41 = 0.0
#            asdf1 = None
            eqs = [fd_abs(elem) for elem in p.user.f]
            asdf1 = fd_sum(eqs)
            
            # temporary change, TODO: rework it
            asdf1.resolveSchedule = asdf1._getDep()
            p._objDep = asdf1.Dep & p.freeVarsSet
            
            # TODO: check it, for reducing calculations
            #C.update([elem == 0 for elem in p.user.f])
        elif isMOP:
            asdf1 = p.user.f
            p._objDep = set()
            p._objDep.update(*[_f.Dep for _f in asdf1])
            p._objDep &= p.freeVarsSet
            Solutions.F = []
            if point(p.x0).isFeas(altLinInEq=False):
                Solutions.solutions.append(p.x0.copy())
                Solutions.coords = asarray(Solutions.solutions)
                Solutions.F.append(p.f(p.x0))
                p._solutions = Solutions
        elif not isODE:
            asdf1 = p.user.f[0]
            p._objDep = asdf1.Dep & p.freeVarsSet
            #if p.fOpt is not None:  fOpt = p.fOpt
            if p.goal in ('max', 'maximum'):
                asdf1 = -asdf1
                if p.fOpt is not None:
                    p.fOpt = -p.fOpt
            
            if fStart is not None and fStart < r40: 
                r41 = fStart
                
            for X0 in [point(xRecord), point(p.x0)]:
                if X0.isFeas(altLinInEq=False) and X0.f() < r40:
                    r40 = X0.f()

            if p.isFeas(p.x0):
                tmp = asdf1(p._x0)
                if  tmp < r41:
                    r41 = tmp
                
            if p.fOpt is not None:
                if p.fOpt > r41:
                    p.warn('user-provided fOpt seems to be incorrect, ')
                r41 = p.fOpt
        else:# ODE
            # TODO: implement p._objDep = asdf1.Dep & p.freeVarsSet here
            assert 0, 'unimplemented yet'

#        if isSNLE:
#            if self.dataHandling == 'raw':
#                p.pWarn('''
#                    this interalg data handling approach ("%s") 
#                    is unimplemented for SNLE yet, dropping to "sorted"'''%self.dataHandling)
#            
#            # handles 'auto' as well
#            self.dataHandling ='sorted'

        #from FuncDesigner.ooFun import BooleanOOFun, SmoothFDConstraint
        
        if p.hasLogicalConstraints:
            if self.dataHandling == 'sorted': 
                p.warn("interalg: for general logical constraints only dataHandling='raw' mode works")
            self.dataHandling = 'raw'
        
        if not isMOP and not p.hasLogicalConstraints:
            p._isOnlyBoxBounded = p.__isNoMoreThanBoxBounded__() 

        if self.dataHandling == 'auto':
            if isIP or isODE:
                self.dataHandling = 'sorted'
            elif isMOP or p.hasLogicalConstraints:
                self.dataHandling = 'raw'
            else:
                r = p.user.f[0].interval(domain, self.dataType)
                M = np.max((np.max(np.atleast_1d(np.abs(r.lb))), np.max(np.atleast_1d(np.abs(r.ub)))))
                for (c, func, lb, ub, tol) in p._FD.nonBoxCons:#[Elem[1] for Elem in p._FD.nonBoxCons]:

                    # !!!!!!!!!!!!!!!!!!!! check it - mb 2nd condition is incorrect
                    #if isinstance(c, BooleanOOFun) and not isinstance(c, SmoothFDConstraint): continue
                    if c.discrete:#hasattr(c,'_unnamedBooleanOOFunNumber'):
                        continue
                    
                    r = func.interval(domain, self.dataType)
                    M = np.max((M, np.max(np.atleast_1d(np.abs(r.lb)))))
                    M = np.max((M, np.max(np.atleast_1d(np.abs(r.ub)))))
                
                self.dataHandling = 'raw' if M < 1e5 else 'sorted'
                
                # TODO: is it required yet?
                if asdf1.isUncycled and p._isOnlyBoxBounded and np.all(np.isfinite(p.user.f[0].interval(domain).lb)):
                    #maxNodes = 1
                    self.dataHandling = 'sorted'
                if p.iprint >= 0:
                    p.disp('interalg parameter dataHandling has been autoselected to "%s"' % self.dataHandling)

            #self.dataHandling = 'sorted' if isIP or (p.__isNoMoreThanBoxBounded__() and n < 50) else 'raw'
        
        if self.dataHandling == 'sorted':
            self._constraints_reduction = False
        
        self.maxActiveNodes = int(self.maxActiveNodes)
#        if self.maxActiveNodes < 2:
#            p.warn('maxActiveNodes should be at least 2 while you have provided %d. Setting it to 2.' % self.maxActiveNodes)
        self.maxNodes = int(self.maxNodes)
        
        g = np.inf
        
#        A = getattr(p, 'A', None)
#        if A is not None:
#            p.A2 = [line for line in p.A]
#            p.b2 = p.b.tolist()
        
        # TODO: improve C & C0, omit unused data, check for their processing in openopt kernel
        _C = p._FD.nonBoxConsWithTolShift
        _C0 = p._FD.nonBoxCons
        
        if isSNLE:
# !!!!!!!!!!!!!!!! TODO: connect linear constraints to Aeq & beq matrices
            _C += [(elem==0, elem, -(elem.tol if elem.tol != 0 else p.ftol), (elem.tol if elem.tol != 0 else p.ftol)) for elem in p.user.f]
            _C0 += [(e[0], e[1], 0, 0, (e[1].tol if e[1].tol != 0 else p.ftol)) for e in _C]
            

        if 1:
            C = SortedDict((elem[0]._id, elem) for elem in _C)
            C0 = SortedDict((elem[0]._id, elem) for elem in _C0)        
        else:
            C = dict((elem[0]._id, elem) for elem in _C)
            C0 = dict((elem[0]._id, elem) for elem in _C0)        
        
        p._constraints_dep_dict = dict((k, elem[1].Dep & p.freeVarsSet) for k, elem in C.items())
        
#        if not use_saved:

        p._constraints_counter = dict((c_id, 0) for c_id in C0.keys())
        
        # TODO: further develoopment (add triggering cons data while point/cs computation)
        activeCons = [ActiveConstraintsEntry(list(C0.keys()))]
        activeCons[0].triggering_info = {}
        
        
#        else:
##            C, C0 = _C, _C0
#            activeCons = [[elem[0]._id for elem in _C0]]
#            p._constraints_counter = dict()
        
#        C.sort(key=lambda elem: elem[0]._id)
#        C0.sort(key=lambda elem: elem[0]._id) # we could use sort order from prev sort but computation time difference is insufficient
        
#        c_id = dict((elem[0]._id, j) for j, elem in enumerate(C0))
        
        # TODO: mb use list or ndarray instead
        
        
        # TODO: hanlde fixed variables here
        varTols = p.variableTolerances
        if Solutions.maxNum != 1:
            if not isSNLE:
                p.err('''
                "search several solutions" mode is unimplemented
                for the prob type %s yet''' % p.probType)
            if any(varTols == 0):
                p.err('''
                for the mode "search all solutions" 
                you have to provide all non-zero tolerances 
                for each variable (oovar)
                ''')
            
        pnc = 0
        an = []
        maxNodes = self.maxNodes
        
        # TODO: change for constrained probs
        _s = atleast_1d(inf)
        
        if isODE or (isIP and p.n == 1):
            from interalgODE import interalg_ODE_routine
            interalg_ODE_routine(p, self)
            return
        
        #_in = np.array([], object)
        _in = []
        
#        print('p.iter:', p.iter)
        while 1:
            
            assert len(activeCons) == _s.size
#            p.iprint=1
            
            if len(C0) != 0: # SNLE also can have constraints
                if use_saved:
                    y, e, nlhc, residual, definiteRange, indT, \
                    _s =\
                    processConstraints(_C0, y, e, _s, p, dataType)
                else:
                    y, e, nlhc, residual, definiteRange, indT, \
                    _s, activeCons =\
                    processConstraints(C, C0, y, e, _s, _in, p, dataType, activeCons)
            else:
                residual, definiteRange, indT = None, True, None
                nlhc = np.empty((y.shape[0], 2*y.shape[1]), np.float64)
                nlhc.fill(self._constraintInactiveValue)
#            a = np.array([-1.26291003e-02,  3.01864599e-02,   1.89144034e-03,  -2.78295240e-04, -2.16195926e-05,   7.54278724e-04,  -4.66201550e-04,  -4.93097371e-05, 3.18665265e-04,  -7.38596373e-04])
#            print('-', p.iter, hasPoint(y, e, a), pointInd(y, e, a))

            assert len(activeCons) == _s.size == y.shape[0]
            
            if y.size != 0:
                an, g, fo, _s, Solutions, xRecord, r41, r40 = \
                pb(p, nlhc, residual, definiteRange, y, e, vv, asdf1, C, r40, g, \
                             nNodes, r41, fTol, Solutions, varTols, _in, \
                             dataType, maxNodes, _s, indT, xRecord, activeCons)
                if _s is None:
                    break 
                
            else:
                an = _in
                # TODO: mb remove fo recalculation from here
                fo = 0.0 if isSNLE or isMOP else \
                PythonMin(r41, \
                          r40 - (fTol if Solutions.maxNum == 1 else 0.0))
            # TODO: check it                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
            pnc = PythonMax(pnc, 
                                    len(np.atleast_1d(an) if type(an) == np.ndarray else an))
            
#            assert len(activeCons) == _s.size == y.shape[0]
            
            
            y, e, _in, _s, activeCons = \
                func12(an, self.maxActiveNodes, p, Solutions, vv, varTols, np.inf if isIP else fo)
            
            nActiveNodes.append(y.shape[0]//2)
            if y.size == 0: 
                if len(Solutions.coords) > 1:
                    p.istop, p.msg = 1001, 'all solutions have been obtained'
                else:
                    p.istop, p.msg = 1000, 'solution has been obtained'
                break            
        '''                        ^^^ End of main cycle ^^^                        '''
            
        if not isSNLE and not isIP and not isMOP:
            p_xk = p.point(p.xk)
            p.iterfcn(p._bestPoint if p._bestPoint.betterThan(p_xk) else p_xk)
            
            if isscalar(r40):
                p.extras['r40'] = r40
            if isscalar(r41):
                p.extras['r41'] = r41
            if isscalar(g):
                p.extras['g'] = g
            

        
        ff = p.fk # ff may be not assigned yet
#        ff = p._bestPoint.f()
#        p.xk = p._bestPoint.x
        if isIP: 
            p.xk = array([np.nan]*p.n)
            p.rk = p._residual
            p.fk = p._F
        
        isFeas = len(Solutions.F) != 0 if isMOP else p.isFeas(p.xk) if not isIP else p.rk < fTol
        
        if not isFeas and p.istop > 0:
            p.istop, p.msg = -1000, 'no feasible solution has been obtained'
        
        if isSNLE or isMOP:
            lf = inf
        else:
            lf = array([t.key for t in an])#.flatten()
            if lf.size != 0:
                g = nanmin((nanmin(lf), g))

        if not isMOP:
            AbsVal = PythonMin(abs(ff), abs(g))
            cond_rTol = ff - g <= p.rTol * AbsVal
            rtol_accuracy = (ff - g) / AbsVal 
            if AbsVal == 0: rtol_accuracy = 100.0 
            
            cond_fTol = ff - g < fTol
            
            p.extras['isRequiredPrecisionReached'] = \
            True if(cond_fTol or cond_rTol) and isFeas else False
            # and (k is False or (isSNLE and (p._nObtainedSolutions >= maxSolutions or maxSolutions==1))) 

        if not isMOP and not p.extras['isRequiredPrecisionReached'] and p.istop > 0:
            p.istop = -1
            p.msg = 'required precision is not guarantied' if p.probType not in ('NLP', 'GLP', 'NSP') else\
            'required precision (fTol = %g, rTol = %g) is not guarantied' % (fTol, p.rTol)
            
        # TODO: simplify it
        if not isMOP:
            tmp = [nanmin(np.hstack((ff, g, lf))), np.asscalar(np.array(ff))]
            if p.goal in ['max', 'maximum']: tmp = (-tmp[1], -tmp[0])
            p.extras['extremumBounds'] = tmp if not isIP else 'unimplemented for IP yet'
        
        if isMOP or p.maxSolutions != 1:
            tmp = [p._vector2point(s) for s in Solutions.coords]
            if isMOP:
                # TODO: mb rework it
                from openopt.kernel.MOP import MOPsolutions 
                tmp = MOPsolutions(tmp)
            p.solutions = tmp
        if isMOP:
            for i, s in enumerate(p.solutions):
                s.useAsMutable = True
                for v, val in s.items():
                    if v.fields != ():
                        s[v] = dict((field, v.aux_domain[int(val)][j]) for j, field in enumerate(v.fields))
                for j, goal in enumerate(p.user.f):
                    s[goal] = Solutions.F[i][j]
                s.useAsMutable = False
            p.solutions.values = np.asarray(Solutions.F)
            p.solutions.coords = Solutions.coords
        if isSNLE and p.maxSolutions != 1:
            for v in p._stringVars:
                for elem in r.solutions:
                    elem.useAsMutable = True
                    elem[v] = v.aux_domain[elem[v]]
                    elem.useAsMutable = False
        if p.iprint >= 0 and not isMOP:
            s = 'Solution with required tolerance %0.1e \n is%s guarantied' \
            %(fTol, '' if p.extras['isRequiredPrecisionReached'] else ' NOT') \
            if p.probType not in ('NLP', 'GLP', 'NSP', 'MINLP')\
            else 'Solution with required tolerances fTol = %g, rTol = %g \n is%s guarantied' \
            %(fTol, p.rTol, '' if p.extras['isRequiredPrecisionReached'] else ' NOT')
            
            if not isIP and p.maxSolutions == 1:
                s += '\n obtained precision: %0.1e   relative accuracy: %0.1g (%0.1g%%)' % (np.abs(tmp[1]-tmp[0]), rtol_accuracy, 100*rtol_accuracy)
            if not p.extras['isRequiredPrecisionReached'] and pnc == self.maxNodes: s += '\nincrease maxNodes (current value %d)' % self.maxNodes
            p.info(s)

class Solution:
    pass
    

import sys
class ActiveConstraintsEntry(list):
    def copy(self):
        if sys.version_info > (3, 3):
            tmp = ActiveConstraintsEntry(list.copy(self))
            tmp.triggering_info = dict((k, w.copy()) for k, w in self.triggering_info.items())
        else:
            tmp = ActiveConstraintsEntry(self[:])
            tmp.triggering_info = dict((k, w[:]) for k, w in self.triggering_info.items())
        return tmp
