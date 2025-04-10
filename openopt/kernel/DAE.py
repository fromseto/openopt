from numpy import linspace
from FuncDesigner import *
#N = 10
N = 100
t = linspace(0, 1, N)

if 0:
    x = oovars(N)
    y = oovars(N)
else:
    x = oovar('x')
    y = oovar('y')

#eq1 = ((1- 0.5*sin(1e-4*y) + 0.5*d(x, t) + 0.4*cos(d(y, t)) - x**2-y**2)[1:N] == 0)(tol=1e-5)
#eq2 = ((t+y-0.1*y**2+0.01*sin(y) + 0.5*cos(0.5*d(y, t)*t+1))[0:N-1] == 0)(tol=1e-5) # == sin(t)
eq1 = ((- sin(y) + 50*cos(d(x, t)+100))[1:N] == 0)(tol=1e-5)
eq2 = ((x + sin(d(y, t)))[0:N-1] == 0)(tol=1e-5)
#bc1 = d(x, t)[0] == 0.1 
bc1 = x[0] == 0.1 
bc2 = y[N-1] == 0.3
eqs = [eq1, eq2, bc1, bc2]
startPoint = {x:[0]*N, y:[0]*N}

#eq1 = ((d(d(x, t), t) +x)[1:N] == 0)(tol = 1e-5)
#eq1 = ((d2(x, t) +x)[1:N] == 0)(tol = 1e-5)
mu = 0.5
A = 1
w = 60
eq1 = ((d2(x, t) - mu * (1-x**2)*d(x, t) + x - A*sin(w*t))[1:N] == 0)(tol = 1e-4)
bc1 = x[-1] == 0.1 
eqs = [eq1, bc1]
startPoint = {x:[0]*N}

p = dae(eqs, t)
r = p.solve(solver= 'nssolve')
r.plot(x)
