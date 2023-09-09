# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:55:25 2023

@author: Timofey Chernobrovkin (412642) Academic group J4133c
"""

import numpy as np
import timeit
import matplotlib.pyplot as plt
import random
from statistics import mean

n = 2000
k = 5

# I Generate an n-dimensional random vector v = [v1, v2, … , vn)] with non-negative elements

def timespent(f, k, n):
    all_time = []
    for i in range(n):
        t = 0
        for j in range(k):
            a = np.random.randint(0, 100, i)
            t1 = timeit.default_timer()
            f(a)
            t2 = timeit.default_timer()
            t += (t2 - t1)
        all_time.append(t / k)
    return all_time

###================================================================###
###==================== 1) constant function ======================###
###================================================================###
def constant(input):
    return 1

time_con = timespent(constant, k, n)

plt.figure(figsize=(10,5), dpi=300)
plt.title('constant', fontsize=14)
plt.plot(time_con, color='black', label='empyrical')
plt.axhline(y=min(time_con[:15]), color='b', label='theoretical')
plt.ylabel('time, s.', fontsize=14)
plt.xlabel('n', fontsize=14)
plt.legend()
plt.show()

###================================================================###
###=================== 2) the sum of elements =====================###
###================================================================###
def summ(input):
    s = 0
    for i in input:
        s += i
    return s

time_sum = timespent(summ, k, n)

plt.figure(figsize=(10,5), dpi=300)
plt.title('sum', fontsize=14)
plt.plot(time_sum, color='black', label='empyrical')

x = [0, 2000]
y = [0, min(time_sum[-7:])]
plt.plot(x, y, color='b', label='theoretical')

plt.ylabel('time, s.', fontsize=14)
plt.xlabel('n', fontsize=14)
plt.legend()
plt.show()

###================================================================###
###================= 3) the product of elements ===================###
###================================================================###

#3. The product of elements
def product(input):
    p = 1
    for i in input:
        p *= i
    return p

time_prod = timespent(product, k, n)

plt.figure(figsize=(10,5), dpi=300)
plt.title('prod', fontsize=14)
plt.plot(time_prod, color='black', label='empyrical')

x = [0, 2000]
y = [0, min(time_prod[-10:])]
plt.plot(x, y, color='b', label='theoretical')

plt.ylabel('time, s.', fontsize=14)
plt.xlabel('n', fontsize=14)
plt.legend()
plt.show()

###================================================================###
###==================== 4) polynomial direct ======================###
###================================================================###

def polynominal_direct(input):
    x = 1.5
    r = 0
    for i in range(len(input)):
        r += input[i] * (x ** i)
    return r

time_poly_dir = timespent(polynominal_direct, k, 1500)

plt.figure(figsize=(10,5), dpi=300)
plt.title('a direct calculation polynomial function', fontsize=14)
plt.plot(time_poly_dir, color='black', label='empyrical')

x = [0, 1500]
y = [0, min(time_poly_dir[-10:])]
plt.plot(x, y, color='b', label='theoretical')

plt.ylabel('time, s.', fontsize=14)
plt.xlabel('n', fontsize=14)
plt.legend()
plt.show()

###================================================================###
###=============== 4) polynomial Horner’s method  =================###
###================================================================###

def polynominal_horny(input): 
    x = 1.5
    r = 0
    for i in range(len(input), 0, -1):
        r = input[i - 1] + x * r
    return r

time_poly_horner = timespent(polynominal_horny, k, 1500)

plt.figure(figsize=(10,5), dpi=300)
plt.title('polynomial horner', fontsize=14)
plt.plot(time_poly_horner, color='black', label='empyrical')

x = [0, 1500]
y = [0, min(time_poly_horner[-10:])]
plt.plot(x, y, color='b', label='theoretical')

plt.ylabel('time, s.', fontsize=14)
plt.xlabel('n', fontsize=14)
plt.legend()
plt.show()

###================================================================###
###======================= 5) Bubble Sort  ========================###
###================================================================###

def bubble(input):
    for i in range(0,len(input)-1): 
        for j in range(len(input)-1): 
            if(input[j]>input[j+1]): 
                temp = input[j] 
                input[j] = input[j+1] 
                input[j+1] = temp 
    return input

time_bubble = timespent(bubble, k, 500)

plt.figure(figsize=(10,5), dpi=300)
plt.title('Bubble Sort', fontsize=14)
plt.plot(time_bubble, color='black', label='empyrical')

from scipy.interpolate import interp1d
x = [time_bubble.index(time_bubble[0]), time_bubble.index(time_bubble[100]), time_bubble.index(time_bubble[300]), time_bubble.index(time_bubble[400]), time_bubble.index(time_bubble[499])]
y = [min(time_bubble[:10]), min(time_bubble[85:110]), min(time_bubble[285:310]), min(time_bubble[385:410]), min(time_bubble[485:499])]
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')
xnew = np.linspace(time_bubble.index(time_bubble[0]), time_bubble.index(time_bubble[499]))
plt.plot(xnew, f2(xnew), color='b', label='theoretical')

plt.ylabel('time, s.', fontsize=14)
plt.xlabel('n', fontsize=14)
plt.legend()
plt.show()

###================================================================###
###======================= 6) Quick Sort  =========================###
###================================================================###

def QuickSort(array):

    less = []
    equal = []
    greater = []

    if len(array) > 1:
        pivot = array[0]
        for x in array:
            if x < pivot:
                less.append(x)
            elif x == pivot:
                equal.append(x)
            elif x > pivot:
                greater.append(x)
        return QuickSort(less)+equal+QuickSort(greater)
    else:
        return array

time_quick = timespent(QuickSort, k, n)

plt.figure(figsize=(10,5), dpi=300)
plt.title('Quick Sort', fontsize=14)
plt.plot(time_quick, color='black', label='empyrical')

x = [0, n]
y = [0, min(time_quick[-10:])]
plt.plot(x, y, color='b', label='theoretical')

plt.ylabel('time, s.', fontsize=14)
plt.xlabel('n', fontsize=14)
plt.legend()
plt.show()

###================================================================###
###======================== 7) Timsort  ===========================###
###================================================================###

def timsort(input):
    return np.sort(input, kind='stable')

time_tim_sort = timespent(timsort, k, n)

plt.figure(figsize=(10,5), dpi=300)
plt.title('Timsort', fontsize=14)
plt.plot(time_tim_sort, color='black', label='empyrical')

x = [0, n]
y = [0, min(time_tim_sort[-5:])]
plt.plot(x, y, color='b', label='theoretical')

plt.ylabel('time, s.', fontsize=14)
plt.xlabel('n', fontsize=14)
plt.legend()
plt.show()

# II. Generate random matrices A and B of size n × n with non-negative elements. 
# Find the usual matrix product for A and B.

time = []
for i in range(0, 500):
        t = 0
        for j in range(k):
            A = np.random.randint (0, 100, (i, i))
            B = np.random.randint (0, 100, (i, i))
            t1 = timeit.default_timer()
            x = A.dot(B)
            t2 = timeit.default_timer()
            t += (t2 - t1)
        time.append(t / j)

plt.figure(figsize=(10,5), dpi=300)
plt.title('Matrix', fontsize=14)
plt.plot(time, color='black', label='empyrical')

x = [time.index(time[0]), time.index(time[100]), time.index(time[300]), time.index(time[400]), time.index(time[499])]
y = [min(time[:10]), min(time[90:110]), min(time[290:310]), min(time[390:410]), min(time[485:499])]
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')
xnew = np.linspace(time.index(time[0]), time.index(time[499]))
plt.plot(xnew, f2(xnew), color='b', label='theoretical')

plt.ylabel('time, s.', fontsize=14)
plt.xlabel('n', fontsize=14)
plt.legend()
plt.show()
