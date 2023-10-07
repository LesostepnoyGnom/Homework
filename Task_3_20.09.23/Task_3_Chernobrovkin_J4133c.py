# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:42:39 2023

@author: Timofei Chernobrovkin (412642) group J4133c
"""

import numpy as np
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy import optimize

while True:
    alf = random.random()
    bet = random.random()
    if alf != 0 and bet != 0 and alf != bet:
        break

k = 100
x = []
y = []

for i in range(k+1):
    d = random.gauss(0.5, 0.125)
    
    x.append(i / 100)
    y.append(alf * x[i] + bet + d)
x = np.array(x)
y = np.array(y)
plt.scatter(x, y)

###================================================================###
###=========== Gradient Descent (linear approximant) ==============###
###================================================================###
n = 0
f_clc = 0

w, b = 0, 0

num = 4000
lr = 0.01
e = 0.001

for i in range(num+1):
    d1dw = 0
    d1db = 0
    N = x.shape[0]
    
    n+=1
    
    for xi, yi in zip(x, y):
        d1dw += 2 * (w * xi + b - yi) * xi
        d1db += 2 * (w * xi + b - yi)
        
        
    
    w = w - lr * (1 / N) * d1dw
    b = b - lr * (1 / N) * d1db
    
    f_clc += 2
    
    # if abs(w - w2) < e and abs(b - b2) < e:
    #     break
    # w = w2
    # b = b2
GD_lin = w * x + b
fig, axs = plt.subplots(figsize = (10, 8), dpi=300)
axs.scatter(x, y, label="Data")
axs.plot(x, w * x + b, 'r', label="Gradient Descent")
axs.set_title('linear approximation')
plt.legend(fontsize=14)

print('Gradient Descent (linear approximant): [a,b, f-calculations, N of iterations] =',[w, b, f_clc, n])
###================================================================###
###========== Gradient Descent (rational approximant) =============###
###================================================================###

n = 0
f_clc = 0

w, b = 0, 0

num = 1000
lr = 0.01
e = 0.001

lst1 = []
lst2 = []

for i in range(num+1):
    d1dw = 0
    d1db = 0
    N = x.shape[0]
    
    n+=1
    
    for xi, yi in zip(x, y):
        d1dw += 2 * (w / (1 + b * xi) - yi) / (1 + b * xi)
        d1db += 2 * (w / (1 + b * xi) - yi) * (- xi * w / (1 + b * xi) ** 2)
    
    w = w - lr * (1 / N) * d1dw
    b = b - lr * (1 / N) * d1db
    
    f_clc += 2
    
    # if abs(w - w2) < e and abs(b - b2) < e:
    #     break
    # w = w2
    # b = b2
GD_rat = w / (1 + b * x)
fig, axs = plt.subplots(figsize = (10, 8), dpi=300)
axs.scatter(x, y, label="Data")
axs.plot(x, w / (1 + b * x), 'r', label="Gradient Descent") 
axs.set_title('rational approximation')
plt.legend(fontsize=14)

print('Gradient Descent (rational approximant): [a,b, f-calculations, N of iterations] =',[w, b, f_clc, n])

###================================================================###
###================== Conjugate Gradient Descent ==================###
###================================================================###

def linear(wb):
    w, b = wb
    return np.sum((w * x + b - y) ** 2, axis=0)

def rational(wb):
    w, b = wb
    return np.sum((w / (1 + b * x) - y) ** 2, axis=0)

fig, axs = plt.subplots(figsize = (10, 8), dpi=300)
print('Conjugate Gradient Descent (linear approximant)')
CGD = minimize(linear, [1., 1.], method='CG', options={'xtol':1e-3, 'disp':True})
w, b = CGD.x
axs.scatter(x, y, label="Data")
CGD_lin = w * x + b
axs.plot(x, w * x + b, 'r', label="Conjugate Gradient Descent")
axs.set_title('linear approximation')
plt.legend(fontsize=14)

print('Conjugate Gradient Descent (linear approximant): [a,b] =',[w, b])

fig, axs = plt.subplots(figsize = (10, 8), dpi=300)
print('Conjugate Gradient Descent (rational approximant)')
CGD = minimize(rational, [1., -0.5], method='CG', options={'xtol':1e-3, 'disp':True})
w, b = CGD.x
axs.scatter(x, y, label="Data")
CGD_rat = CGD.x
axs.plot(x, w / (1 + b * x), 'r', label="Conjugate Gradient Descent")
axs.set_title('rational approximation')
plt.legend(fontsize=14)

print('Conjugate Gradient Descent (rational approximant): [a,b] =',[w, b])

###================================================================###
###============= Newton’s method (linear approximant) =============###
###================================================================###

def linear(wb):
    w, b = wb
    return np.sum((w * x + b - y) ** 2, axis=0)

def d_linear(wb):
    w, b = wb
    return np.array([np.sum(2 * x * (b + w * x - y)), np.sum(2 * (b + w * x - y))])

def hess_lin(wb):
    w, b = wb
    hess = np.ones([2,2])
    hess[0,0] = np.sum(2 * x**2)
    hess[0,1] = np.sum(2 * x)
    hess[1,0] = np.sum(2 * x)
    hess[1,1] = (2)
    return hess

fig, axs = plt.subplots(figsize = (10, 8), dpi=300)
newton = minimize(linear, [1.0, 1.0], method='Newton-CG', jac=d_linear, hess=hess_lin, options={'xtol': 1e-3, 'disp':True})
w, b = newton.x
axs.scatter(x, y, label="Data")
Newton_linear = newton.x
axs.plot(x, w * x + b, 'r', label="Newton’s method")
axs.set_title('linear approximant')
plt.legend(fontsize=14)

print(f'Newton’s method (linear approximant): {w, b}, real values: {alf, bet}')

###================================================================###
###============ Newton’s method (rational approximant) ============###
###================================================================###
def rational(wb):
    w, b = wb
    return np.sum((w / (1 + b * x) - y) ** 2, axis=0)

def d_rational(wb):
    w, b = wb
    return np.array([np.sum((w/(1+b*x) - y)*2/(1+b*x)), np.sum(2*w*x/(1+b*x)**2 * (w/(1+b*x)-y))])

def hess_rat(wb):
    w, b = wb
    hess = np.ones([2,2])
    hess[0,0] = np.sum(2 / (1+b*x)**2)
    hess[0,1] = np.sum(-2*w*x/(1+b*x)**3-2*x*(w/(1+b*x)-y) / (1+b*x)**3)
    hess[1,0] = np.sum(-2*w*x/(1+b*x)**3-2*x*(w/(1+b*x)-y) / (1+b*x)**3)
    hess[1,1] = np.sum(2 * w**2 * x**2 / (1+b*x)**4 * 4 * w * x**2 * (w/(1+b*x)-y) / (1+b*x)**3)
    return hess

fig, axs = plt.subplots(figsize = (10, 8), dpi=300)
newton = minimize(rational, [1.1, -0.5], method='Newton-CG', jac=d_rational, hess=hess_rat, options={'xtol': 1e-3, 'disp':True})
w, b = newton.x

axs.scatter(x, y, label="Data")
Newton_rat = newton.x
axs.plot(x, w / (1 + b * x), 'r', label="Newton’s method")
axs.set_title('rational approximant')
plt.legend(fontsize=14)

print(f'Newton’s method (rational approximant): {w, b}, real values: {alf, bet}')

###================================================================###
###=========== Levenberg-Marquardt (linear approximant) ===========###
###================================================================###
def linear(wb):
    w, b = wb
    return (w * x + b - y) ** 2

fig, axs = plt.subplots(figsize = (10, 8), dpi=300)
lma = optimize.least_squares(linear, [1., 1.], method="lm", xtol=1e-3)
w, b = lma.x
axs.scatter(x, y, label="Data")
lma_linear = lma.x
axs.plot(x, w * x + b, 'r', label="Levenberg-Marquardt")
axs.set_title('linear approximant')
plt.legend(fontsize=14)

print(f'Levenberg-Marquardt (linear approximant): {w, b}, real values: {alf, bet}')

###================================================================###
###========== Levenberg-Marquardt (rational approximant) ==========###
###================================================================###
def rational(wb):
    w, b = wb
    return (w / (1 + b * x) - y) ** 2

lma = optimize.least_squares(rational, [1., -0.5], method="lm", xtol=1e-3)
w, b = lma.x

fig, axs = plt.subplots(figsize = (10, 8), dpi=300)
axs.scatter(x, y, label="Data")
lma_rat = lma.x

axs.plot(x, w / (1 + b * x), 'g', label="Levenberg-Marquardt") 
axs.set_title('rational approximant')
plt.legend(fontsize=14)

print(f'Levenberg-Marquardt (rational approximant): {w, b}, real values: {alf, bet}')

'''
               ###======================================###
    ###============================================================###
###======================================================================###
    ###============== comparison with the second task =============###
###======================================================================###
    ###============================================================###
               ###======================================###
'''

###================================================================###
###=========== exhaustive search (linear approximant) =============###
###================================================================###

lst = []
for a in range(0, 1001):
    a /= 1000
    for b in range(0, 1001):
        b /= 1000
        D = 0
        for i in range(0, k+1):
            D += (a * x[i] + b - y[i]) ** 2
        lst.append([D, a, b])

minimum = min(x[0] for x in lst)
for lst2 in lst:
    if lst2[0] == minimum:
        a = lst2[1]
        b = lst2[2]
        break

p = []
for i in range(len(x)):
    p.append(x[i]*a + b)
exh_lin = p
fig, axs = plt.subplots(figsize=(10,8), dpi=300)
plt.title("Linear", fontsize=14)
axs.scatter(x, y, label="Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x, p, 'g', label="Exhaustive")
plt.legend(fontsize=14)

print('exhaustive search (linear approximant): [a,b, f-calculations, N of iterations] =',[a, b, 1000 ** 2, 1000 ** 2])

###================================================================###
###========== exhaustive search (rational approximant) ============###
###================================================================###
list2 = []
for a in range(0, 1000):
    a = a / 1000
    for b in range(0, 1000):
        b = -b / 1000
        s = 0
        for k in range(0, 101):
            s += (a / (1 + b * x[k]) - y[k]) ** 2
        list2.append([s, a, b])

minimum = min(x[0] for x in list2)
for sublist in list2:
    if sublist[0] == minimum:
        a = sublist[1]
        b = sublist[2]
        break
    
p = []
for i in range(len(x)):
    p.append(a/(1 + b * x[i]))
exh_rat = p
fig, axs = plt.subplots(figsize=(10,8), dpi=300)
plt.title("Rational", fontsize=14)
axs.scatter(x, y, label="Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x, p, 'g', label="Exhaustive")
plt.legend(fontsize=14)

print('exhaustive search (rational approximant): [a,b, f-calculations, N of iterations] =',[a, b, 1000 ** 2, 1000 ** 2])

##================================================================###
##=================== Gauss (linear function) ====================###
##================================================================###

a1 = 0.02
b1 = 0.02
b1_2 = 2500
N = 0
f_clc = 0

while True:
    N += 2
    lst1 = []
    for a in range(0, 1001):
        f_clc += 1
        a /= 1000
        D = 0
        for i in range(0, k+1):
            D += (a * x[i] + b1 - y[i]) ** 2
        lst1.append(D)
    a1_2 = lst1.index(min(lst1)) / 1000
    if abs(a1 - a1_2) < e and abs(b1 - b1_2) < e:
        break
    a1 = a1_2

    lst2 = []
    for b in range(0, 1001):
        f_clc += 1
        b /= 1000
        D = 0
        for i in range(0, k+1):
            D += (a1 * x[i] + b - y[i]) ** 2
        lst2.append(D)
    b1_2 = lst2.index(min(lst2)) / 1000
    if abs(b1 - b1_2) < e and abs(a1 - a1_2) < e:
        break
    b1 = b1_2
    
p = []
for i in range(len(x)):
    p.append(x[i]*a1 + b1)
gauss_lin = p
fig, axs = plt.subplots(figsize=(10,8), dpi=300)
plt.title("Linear", fontsize=14)
axs.scatter(x, y, label="Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x, p, 'g', label="Gauss")
plt.legend(fontsize=14)

print('Gauss (linear function): [a,b, f-calculations, N of iterations] =',[a1, b1, f_clc, N])

###================================================================###
###================ Gauss (rational approximant) ==================###
###================================================================###
a1 = 0.01
b1 = -0.01
b1_2 = 7000
cnt = 0
f_clc = 0
while True:
    cnt += 2
    lst1 = []
    for a in range(0, 1000):
        f_clc += 1
        a /= 1000
        s = 0
        for i in range(0, k+1):
            s += (a / (1 + b1 * x[i]) - y[i]) ** 2
        lst1.append(s)
    a1_2 = lst1.index(min(lst1)) / 1000
    if abs(a1 - a1_2) < e and abs(b1 - b1_2) < e:
        break
    a1 = a1_2
    lst2 = []
    for b in range(0, 1000):
        f_clc += 1
        b = -b/1000
        s = 0
        for i in range(0, k+1):

            s += (a1 / (1 + b * x[i]) - y[i]) ** 2
        lst2.append(s)
    b1_2 = lst2.index(min(lst2)) / -1000
    if abs(b1 - b1_2) < e and abs(a1 - a1_2) < e:
        break
    b1 = b1_2
p = []
for i in range(len(x)):
    p.append(a1/(1 + b1 * x[i]))
gauss_rat = p
print('Gauss  method (rational approximant): a =', a, ', b =', b, ', f-calculations =', f_clc, 'N =', cnt)

fig, axs = plt.subplots(figsize=(10,8), dpi=300)
plt.title("Rational", fontsize=14)
axs.scatter(x, y, label="Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x, p, 'g', label="Gauss")
plt.legend(fontsize=14)

###================================================================###
###================ Nelder-Mead (linear function) =================###
###================================================================###
def fcn(ab):
    a, b = ab
    D = 0
    for i in range(0, k+1):
        D += (a * x[i] + b - y[i]) ** 2
    return D

r = minimize(fcn,[1, 0.5], method='nelder-mead', options={'xatol': e,'disp': True})

p = []
for i in range(len(x)):
    p.append(r.x[0] * x[i] + r.x[1])
NM_lin = p
print ('Nelder-Mead (linear function): [a, b] =', r.x)
fig, axs = plt.subplots(figsize=(10,8), dpi=300)
plt.title("Linear", fontsize=14)
axs.scatter(x, y, label="Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x, p, 'g', label="Nelder-Mead")
plt.legend(fontsize=14)

###================================================================###
###============== Nelder-Mead (rational approximant) ==============###
###================================================================###
def fun(ab):
    a, b = ab
    s = 0
    for k in range(0, 101):
        s += (a / (1 + b * x[k]) - y[k]) ** 2
    return s

result = minimize(fun,[0.3, 0.3], method='nelder-mead', options={'xatol': 0.001,'disp': True})

nm_rat = result.x

p = []
for i in range(len(x)):
    p.append(result.x[0] / (1 + result.x[1] * x[i]))
NM_rat = p
print ('Nelder-Mead (rational approximant): [a, b] =', result.x)
fig, axs = plt.subplots(figsize=(10,8), dpi=300)
plt.title("Rational", fontsize=14)
axs.scatter(x, y, label="Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x, p, 'g', label="Nelder-Mead")
plt.legend(fontsize=14)





plt.figure(figsize=(10,7), dpi=300)
plt.title("Linear", fontsize=14)
plt.plot(x, y, '.b', label="Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x, GD_lin, 'g', label="Gradient Descent")
plt.plot(x, CGD_lin, 'r', label="Conjugate Gradient")
plt.plot(x, Newton_linear[0]*x + Newton_linear[1], 'y', label="Newton")
plt.plot(x, lma_linear[0]*x + lma_linear[1], 'm', label="Levenberg-Marquardt")
plt.plot(x, exh_lin, 'black', label="Exhaustive")
plt.plot(x, gauss_lin, 'orange', label="Gauss")
plt.plot(x, NM_lin, 'lime', label="Nelder-Mead")
plt.legend(fontsize=14)


plt.figure(figsize=(10,7), dpi=300)
plt.title("Rational", fontsize=14)
plt.plot(x, y, '.b', label="Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x, GD_rat, 'g', label="Gradient Descent")
plt.plot(x, CGD_rat[0] /(1 + CGD_rat[1]*x), 'r', label="Conjugate Gradient")
plt.plot(x, Newton_rat[0] / (1 + Newton_rat[1]*x), 'y', label="Newton")
plt.plot(x, lma_rat[0] / (1 + lma_rat[1]*x), 'm', label="Levenberg-Marquardt")
plt.plot(x, exh_rat, 'black', label="Exhaustive")
plt.plot(x, gauss_rat, 'orange', label="Gauss")
plt.plot(x, NM_rat, 'lime', label="Nelder-Mead")
plt.legend(fontsize=14)