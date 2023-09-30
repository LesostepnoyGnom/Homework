# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 12:03:09 2023

@author: Timofei Chernobrovkin (412642) group J4133c
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy import optimize
from pyswarm import pso
import random, copy

k = 1000
x = []
y = []
e = 0.001

def f(x):
    return 1 / (x**2 - 3*x + 2)

for i in range(k+1):
    d = random.gauss(0.5, 0.5)
    
    x.append((3 * i) / 1000)
    
    if f(x[i]) < -100:
        y.append(-100 + d)
    elif -100 <= f(x[i]) and f(x[i]) <= 100:
        y.append(f(x[i]) + d)
    elif f(x[i]) > 100:
        y.append(100 + d)
    else:
        print('ERROR')
x = np.array(x)
y = np.array(y)
fig, axs = plt.subplots(figsize = (10, 8), dpi=300)
plt.scatter(x, y)

def rat_fun(x, a, b, c, d):
    return (a * x + b) / (x ** 2 + c * x + d)

def rational_function(x, a, b, c, d):
    return (a * x + b) / (x ** 2 + c * x + d)

def least_squares(vector, func, x, y):
    a, b, c, d = vector
    return np.sum((func(x, a, b, c, d) - y) ** 2)

def least_squares_lm(vector, func, x, y):
    a, b, c, d = vector
    return func(x, a, b, c, d) - y
     


###================================================================###
###========================= Nelder-Mead ==========================###
###================================================================###

x0 = np.ones(4)

result_nm = minimize(least_squares, x0, method='nelder-mead', args=(rational_function, x, y), options={'xatol': e,'disp': True})

nm_rat = result_nm.x

print ('Nelder-Mead: [a, b, c, d] =', result_nm.x)
plt.figure(figsize=(10,7), dpi=300)
plt.plot(x, y, '.b', label="Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x, rational_function(x, *result_nm.x), 'g', label="Nelder-Mead")
plt.legend(fontsize=14)

###================================================================###
###====================== Levenberg-Marquardt =====================###
###================================================================###

result_lm = optimize.least_squares(least_squares_lm, x0, method='lm', args=(rational_function, x, y), ftol=e)
print ('Levenberg-Marquardt: [a, b, c, d] =', result_lm.x)
plt.figure(figsize=(10,7), dpi=300)
plt.plot(x, y, '.b', label="Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x, rational_function(x, *result_lm.x), 'g', label="Levenberg-Marquardt")
plt.legend(fontsize=14)
###================================================================###
###======================= Particle Swarm =========================###
###================================================================###

lb = np.ones(4) * -1 
ub = np.ones(4) 
xopt, fopt = pso(least_squares, lb, ub, maxiter=k, args=(rational_function, x, y), swarmsize=k, minfunc=e, debug=True)
result_pso = xopt

print ('Particle Swarm: [a, b, c, d] =', result_pso)
plt.figure(figsize=(10,7), dpi=300)
plt.plot(x, y, '.b', label="Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x, rational_function(x, *result_pso), 'g', label="Particle Swarm")
plt.legend(fontsize=14)
print(result_pso)
###================================================================###
###=================== Differential Evolution =====================###
###================================================================###

bounds = np.array([[-2, 2], [-2, 2], [-2, 2], [-2, 2]])
result_de = optimize.differential_evolution(least_squares, bounds, args=(rational_function, x, y), tol=e)

print ('Differential Evolution: [a, b, c, d] =', result_de.x)
plt.figure(figsize=(10,7), dpi=300)
plt.plot(x, y, '.b', label="Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x, rational_function(x, *result_de.x), 'g', label="Differential Evolution")
plt.legend(fontsize=14)

###================================================================###
###================================================================###
###================================================================###

plt.figure(figsize=(10,7), dpi=300)
plt.title("Functions comparison", fontsize=14)
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x, y, '.b', label="Data")
plt.plot(x, rational_function(x, *result_nm.x), 'black', label="Nelder-Mead", linewidth=4)
plt.plot(x, rational_function(x, *result_lm.x), 'red', label="Levenberg-Marquardt")
plt.plot(x, rational_function(x, *result_pso), color='green', label="Particle Swarm")
plt.plot(x, rational_function(x, *result_de.x), color='c', label="Differential Evolution", linestyle = '--')
plt.legend(fontsize=12)

'''
Travelling Salesman Problem
'''
# St. Petersburg, Moskow, Yekaterinburg, Artemovsky, Ulaanbaatar,
# Wuhan, Tampere, Hanoi, Hong Kong, Dresden, Orleans, Lisbon, Norilsk, Kathmandu, Tbilisi
# x = [59.9386, 55.7522, 56.8519, 57.3390, 47.9077, 30.5833, 61.4991, 21.0245, 22.2802, 51.0509, 47.9029, 38.7167, 69.3535, 27.7017, 41.6941]
# y = [30.3141, 37.6156, 60.6122, 61.8783, 106.883, 114.267, 23.7871, 105.841, 114.1653, 13.7383, 1.90389, -9.13333, 88.2027, 85.3206, 44.8337]
# plt.figure(figsize=(16,10), dpi=300)
# plt.scatter(x,y, s = 90, color='black')
fname = 'cities_coordinates15.txt'
xy = np.loadtxt(fname)

x = [item[0] for item in xy]
y = [item[1] for item in xy]
names = ['St. Petersburg', 'Moskow', 'Yekaterinburg', 'Artemovsky', 'Ulaanbaatar', 'Wuhan', 'Tampere', 'Hanoi', 'Hong Kong', 'Dresden', 'Orleans', 'Lisbon', 'Norilsk', 'Kathmandu', 'Tbilisi']
plt.figure(figsize=(16,10), dpi=300)
plt.scatter(x,y, s=90, color='black')
for i in range(len(names)):
    plt.annotate(names[i], xy=(x[i] + 0.5, y[i]+0.2))

def distance(xy1, xy2):
    return np.sqrt(np.sum((xy2 - xy1) ** 2))

def distance_matrix_func(xy):
    distance_matrix = np.ones((xy.shape[0], xy.shape[0]))
    for i, xy1 in enumerate(xy):
        for j, xy2 in enumerate(xy):
            distance_matrix[i, j] = distance(xy1, xy2)
    return distance_matrix

def total_distance(distance_matrix):
    S = 0
    for i in range(14):
        S += distance_matrix[i + 1, i]
    S += distance_matrix[0,14]
    return S

#1st iteration visualization
print('Total distance: ', total_distance(distance_matrix_func(xy)))
plt.figure(figsize=(16,10), dpi=300)
plt.scatter(x,y, s=90, color='black')
for i in range(len(names)):
    plt.annotate(names[i], xy=(x[i] + 0.5, y[i]+0.2))
for i in range(14):
    plt.arrow(xy[i][0],xy[i][1],xy[i+1][0]-xy[i][0],xy[i+1][1]-xy[i][1])
plt.arrow(xy[14][0],xy[14][1],xy[0][0]-xy[14][0],xy[0][1]-xy[14][1])

random.seed(123456)
T = 100000
S_list = []
S = total_distance(distance_matrix_func(xy))
S_list.append(S)
i = 0
while S > 330:
# while T > 0:
    i += 1
    i_swap = random.sample(range(15), 2)
    xy_copy = copy.copy(xy)
    xy_temp = copy.copy(xy_copy[i_swap[0]])
    xy_copy[i_swap[0]] = xy_copy[i_swap[1]]
    xy_copy[i_swap[1]] = xy_temp
    S = total_distance(distance_matrix_func(xy_copy))
    S_list.append(S)
    if S_list[i] < S_list[i - 1]:
        xy = xy_copy
        T = T * 0.999
    else:
        delta = S_list[i] - S_list[i - 1]
        probability = np.exp(-delta / T)
        random_roll = random.uniform(0, 1)
        if random_roll < probability:
            xy = xy_copy
            T = T * 0.999
            
plt.figure(figsize=(16,10), dpi=300)
plt.scatter(x,y, s=90, color='black')
for i in range(len(names)):
    plt.annotate(names[i], xy=(x[i] + 0.5, y[i]+0.2))
for i in range(14):
    plt.arrow(xy[i][0],xy[i][1],xy[i+1][0]-xy[i][0],xy[i+1][1]-xy[i][1])
plt.arrow(xy[14][0],xy[14][1],xy[0][0]-xy[14][0],xy[0][1]-xy[14][1])

print('Optimized (minimal) distance: ', total_distance(distance_matrix_func(xy)))