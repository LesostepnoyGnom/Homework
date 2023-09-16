# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 16:01:00 2023

@author: Chernobrovkin Timofei (412642) group J4133c
"""

'''
I. Use the one-dimensional methods of exhaustive search, dichotomy, and golden
section search to find an approximate (with precision ε=0.001) solution
x:f(x)→min for the following functions and domains:
'''

import numpy as np
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt
###================================================================###
###========== 1. Exhaustive search (brute-force search) ===========###
###================================================================###

e = 0.001
a = 0
b = 1

def f1(x):
    return x**3
def f2(x):
    return abs(x-0.2)
def f3(x):
    return x*np.sin(1/x)

def exhaustive(f, e, a, b):

    n = int((b-a)/e)

    lst = []
    for k in range(0, n+1):
        xk = a + k*((b-a)/n)
        lst.append(f(xk))
    return min(lst), len(lst)

print('f(x) = x**3: x^ =',exhaustive(f1, e, a, b)[0], ', f-calculation =', exhaustive(f1, e, a, b)[1], ', n =',int((b-a)/e))
print('f(x) = |x - 0.2|: x^ =', exhaustive(f2, e, a, b)[0], ', f-calculation =', exhaustive(f2, e, a, b)[1], ', n =',int((b-a)/e))
print('f(x) = x*sin(1/x): x^ =', exhaustive(f3, e, 0.01, b)[0], ', f-calculation =', exhaustive(f3, e, 0.01, b)[1], ', n =',int((b-0.01)/e))
###================================================================###
###===================== 2. Dichotomy method ======================###
###================================================================###
# 2. Dichotomy method

def dih(f, e, a, b):
    d = 0.0005  # let's take a random value which 0 < d < e
    cnt = 0     # counter of iteration
    clc = 0     # counter of f-calculation
    
    while abs(a - b) >= e:
        cnt += 1
        
        x1 = (a + b - d) / 2
        clc += 1
        x2 = (a + b + d) / 2
        clc += 1
        
        if f(x1) <= f(x2):
            b = x2
        else:
            a = x1
            
    return f((a + b) / 2), clc, cnt

print('f(x) = x**3: x^ =', dih(f1, e, a, b)[0], ', f-calculation =', dih(f1, e, a, b)[1], ', n =', dih(f1, e, a, b)[2])
print('f(x) = |x - 0.2|: x^ =', dih(f2, e, a, b)[0], ', f-calculation =', dih(f2, e, a, b)[1], ', n =', dih(f2, e, a, b)[2])
print('f(x) = x*sin(1/x): x^ =', dih(f3, e, 0.01, b)[0], ', f-calculation =', dih(f3, e, 0.01, b)[1], ', n =', dih(f3, e, 0.01, b)[2])
###================================================================###
###==================== Golden section method =====================###
###================================================================###

def gold(f, e, a, b):
    cnt = 0     # counter of iteration
    clc = 0     # counter of f-calculation
    
    x1 = a + ((3 - 5 ** 0.5) / 2) * (b - a)
    clc += 1
    x2 = b + ((5 ** 0.5 - 3) / 2) * (b - a)
    clc += 1
    while abs(a - b) >= e:
        cnt += 1
    
        if f(x1) <= f(x2):
            b = x2
            x2 = x1
            x1 = a + (3 - np.sqrt(5)) / 2 * (b - a)
            clc += 1
        else:
            a = x1
            x1 = x2
            x2 = b + ((5 ** 0.5 - 3) / 2) * (b - a)
            clc += 1
    return f((a + b) / 2), clc, cnt

print('f(x) = x**3: x^ =', gold(f1, e, a, b)[0], ', f-calculation =', gold(f1, e, a, b)[1], ', n =', gold(f1, e, a, b)[2])
print('f(x) = |x - 0.2|: x^ =', gold(f2, e, a, b)[0], ', f-calculation =', gold(f2, e, a, b)[1], ', n =', gold(f2, e, a, b)[2])
print('f(x) = x*sin(1/x): x^ =', gold(f3, e, 0.01, b)[0], ', f-calculation =', gold(f3, e, 0.01, b)[1], ', n =', gold(f3, e, 0.01, b)[2])

'''
II.
'''

while True:
    alf = random.random()
    bet = random.random()
    if alf != 0 and bet != 0 and alf != bet:
        break

k = 100
x = []
y = []

for i in range(k+1):
    d = random.gauss(0.5, 0.5)
    
    x.append(i / 100)
    y.append(alf * x[i] + bet + d)
    
A = np.vstack([x, np.ones(len(x))]).T
a, b = np.linalg.lstsq(A, y, rcond=None)[0]
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

plt.figure(figsize=(10,7), dpi=300)
plt.title("Linear", fontsize=14)
plt.plot(x, y, '.b', label="Data")
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

plt.figure(figsize=(10,7), dpi=300)
plt.title("Rational", fontsize=14)
plt.plot(x, y, '.b', label="Data")
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

plt.figure(figsize=(10,7), dpi=300)
plt.title("Linear", fontsize=14)
plt.plot(x, y, '.b', label="Data")
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
    
print('Gauss  method (rational approximant): a =', a, ', b =', b, ', f-calculations =', f_clc, 'N =', cnt)

plt.figure(figsize=(10,7), dpi=300)
plt.title("Rational", fontsize=14)
plt.plot(x, y, '.b', label="Data")
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

print ('Nelder-Mead (linear function): [a, b] =', r.x)
plt.figure(figsize=(10,7), dpi=300)
plt.title("Linear", fontsize=14)
plt.plot(x, y, '.b', label="Data")
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

print ('Nelder-Mead (rational approximant): [a, b] =', result.x)
plt.figure(figsize=(10,7), dpi=300)
plt.title("Rational", fontsize=14)
plt.plot(x, y, '.b', label="Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x, p, 'g', label="Nelder-Mead")
plt.legend(fontsize=14)
