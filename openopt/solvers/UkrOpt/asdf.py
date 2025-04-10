#def updateBox(box, equations, timeVariable):
#    timeInterval = box[timeVariable]
#    newBox = {timeVariable:timeInterval}
#    dt = timeInterval[1] - timeInterval[0]
#    for v, df in equations.items():
#        r = df.interval(box)
#        lb, ub = box[v]
#        new_lb, new_ub = lb + r.lb * dt, ub + r.ub * dt
#        newBox[v] = (new_lb, new_ub)
#    return newBox

def getNewBounds(equations, initBounds, knownLimits, timeVariable, startTime, endTime):
    deltaT = endTime - startTime 
    newBounds = {}
    L = knownLimits.copy()
    L[timeVariable] = (startTime, endTime)
    for v, df in equations.items():
        r = df.interval(L)
        lb, ub = initBounds[v]
        new_lb, new_ub = lb + r.lb * deltaT, ub + r.ub * deltaT
        newBounds[v] = (new_lb, new_ub)
    return newBounds

from FuncDesigner import *
from numpy import arange, pi
 
# create some variables
x, y, z, t = oovars('x', 'y', 'z', 't')
# or just x, y, z, t = oovars(4)
 
# Python dict of ODEs
#equations = {
#             x: 2*x + cos(3*y-2*z) + exp(5-2*t), # dx/dt
#             y: arcsin(t/5) + 2*x + sinh(2**(-4*t)) + (2+t+sin(z))**(1e-1*(t-sin(x)+cos(y))), # dy/dt
#             z: x + 4*y - 45 - sin(100*t) # dz/dt
#             }
sigma = 0.05#1e-1
ff = lambda x:  exp(-x**2/(2*sigma)) / sqrt(2*pi*sigma) 

equations = {
             x: 0.07 * x,  #+ 3 * sin(4*t+1), 
             y: 0.09 * y,  #+ 0.1 * x,  #+ 0.2*t**2, 
             z: 0.001 * z + 0.009*y + ff(t-0.5)
             }

startPoint = {x: 3, y: 4, z: 5}

timeVariable = t
timeArray = arange(0, 1, 0.01)

from numpy import log, array, hstack
from copy import copy
def solveByInteralg(equations, startPoint, t, timeArray, knownLimits, tol, inittol = 1e-15, timetol = 1e-15, 
                    maxInnerIter=1000):
    assert len(timeArray) == 2, 'unimplemented yet'
    #variables = set(startPoint.keys())
    startTime, endTime = timeArray[0], timeArray[-1]
    EndTime = copy(endTime) # copy for more safety, endTime can be ndarray
    middleTime = endTime    
    initBox = dict([(v, (val, val)) for v, val in startPoint.items()])
    KnownLimits = [knownLimits]
    r = []
    while True:
        #initBox = KnownLimits[-1]
        initBox[t] = (startTime, middleTime)

        #w = log(tol/inittol) / (EndTime - startTime) 
        #th = tol / exp(w * (middleTime - startTime))
        
        th = tol if middleTime == EndTime else 0.5*tol / (tol/inittol) ** ((middleTime - startTime) / (EndTime - startTime))
        
        print 'th:', th
        
        for i in range(maxInnerIter):
            newKnownLimits = getNewBounds(equations, initBox, KnownLimits[-1], timeVariable, startTime, middleTime)
            KnownLimits[-1] = newKnownLimits
            maxRes = max((val[1]-val[0]) for val in newKnownLimits.values())
#            print 'maxRes:', maxRes
            if maxRes <= th:
                break
#        print 'i:', i, 'maxRes:', maxRes, 'th:', th 
        if maxRes > th:
            success = False
            if middleTime - startTime < timetol:
                istop = -20
                msg = 'timetol (%g) has been reached, solution by the interval method seems to be impossible for the task' % timetol
                return istop, msg, r
        else:
            success = True
                
        KnownLimits.append(newKnownLimits)
        if success:
#            newBox = {}
#            initBox.pop(t)
#            for v, val in initBox.items():
#                lb, ub = newKnownLimits[v]
#                newBox[v] = val[0] + lb, val[1] + ub
            initBox = newKnownLimits
            if t in newKnownLimits:
                newKnownLimits.pop(t)
            r.append((startTime, middleTime, newKnownLimits))
            print 'time done:', middleTime
#        print startTime, middleTime, middleTime - startTime
            if middleTime == EndTime:
                newKnownLimits[t] = (startTime, EndTime)
                istop = 1000
                msg = 'solution has been obtained'
                return istop, msg, r
            startTime, middleTime = middleTime, min((startTime + 2 * (middleTime - startTime), EndTime))
            KnownLimits = KnownLimits[:-1]
            inittol = maxRes
        else:
            middleTime = startTime + 0.5 * (middleTime - startTime)

#        print 'new startTime, middleTime:', startTime, middleTime
#        if endTime == startTime:
#            pass
        
   
initBox = {t:(timeArray[0], timeArray[-1])}

knownLimits = {}
for v in [x, y, z]:
    initBox[v] = (-1e-10, 1e-10)
    knownLimits[v] = (-100, 100)
#from FuncDesigner.ooPoint import ooMultiPoint
#box = ooMultiPoint(initBox)

startTime, endTime = timeArray[0], timeArray[-1]

timeArray = (0, 1)
tol = 0.7
from time import time
T = time()
istop, msg, r = solveByInteralg(equations, startPoint, t, timeArray, knownLimits, tol, inittol = 1e-15, 
                                timetol = 1e-2)
print 'done! istop = %d, time elapsed = %0.1f, len(r) = %d' % (istop, time()-T, len(r))
z_bounds = array([elem[2][z] for elem in r])
t_lb_bounds = array([elem[0] for elem in r])
t_ub_bounds = array([elem[1] for elem in r])
from pylab import plot, show
plot(hstack((t_lb_bounds[0], t_ub_bounds)), hstack((startPoint[z], 0.5*z_bounds.sum(axis=1))))
show()

#for i in range(3):
#    newKnownLimits = getNewBounds(equations, initBox, knownLimits, timeVariable, startTime, endTime)
#    knownLimits = newKnownLimits
#    print newKnownLimits
