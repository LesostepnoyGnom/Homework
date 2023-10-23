# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 22:25:26 2023

@author: 1618047
"""

import timeit
import random
import matplotlib.pyplot as plt
import numpy as np

n = 2000
k = 5

def timespent(f, k, n):
    all_time = []
    for i in range(n):
        t = 0
        for j in range(k):
            a = np.random.randint(-100, 100, i)
            t1 = timeit.default_timer()
            f(a)
            t2 = timeit.default_timer()
            t += (t2 - t1)
        all_time.append(t / k)
    return all_time

###================================================================###
###================ The maximum-subarray problem ==================###
###================================================================###

def maxSubArraySum(MyList):
  max_sum = 0
  current_sum = 0

  max_start = 0
  max_end = 0
  current_start = 0
  current_end = 0

  for i in range(len(MyList)): 
    current_sum += MyList[i]
    current_end = i
    if current_sum < 0:
      current_sum = 0
      # Start a new sequence from next element
      current_start = current_end + 1

    if max_sum < current_sum:
      max_sum = current_sum
      max_start = current_start
      max_end = current_end
      
  print("Maximum SubArray is:", max_sum)
  print("Start index of max_Sum:", max_start)
  print("End index of max_Sum:", max_end)

arr = [-2, -3, 4, -1, -2, 5, -3]

maxSubArraySum(arr)

time = timespent(maxSubArraySum, k, n)

plt.figure(figsize=(10,5), dpi=300)
plt.title('The maximum-subarray problem', fontsize=14)
plt.plot(time, color='black', label='empyrical')

x = [0, 2000]
y = [0, min(time[-7:])]
plt.plot(x, y, color='b', label='theoretical', lw=5)

plt.ylabel('time, s.', fontsize=14)
plt.xlabel('n', fontsize=14)
plt.legend()
plt.show()

###================================================================###
###================ An activity-selection problem =================###
###================================================================###

def timespent2(f, k, n):
    all_time = []
    for i in range(n):
        t = 0
        for j in range(k):
            a = np.random.randint(0, 10, i)
            b = np.random.randint(0, 10, i)
            t1 = timeit.default_timer()
            f(a, b)
            t2 = timeit.default_timer()
            t += (t2 - t1)
        all_time.append(t / k)
    return all_time

def activities(s, f):
    n = len(f)
    print("The following activities are selected")
 
    i = 0
    print(i)
    
    for j in range(n):
        if s[j] >= f[i]:
            print(j)
            i = j
            
s = [1, 3, 0, 5, 8, 5]
f = [2, 4, 6, 7, 9, 9]

activities(s, f)

time = timespent2(activities, k, n)

plt.figure(figsize=(10,5), dpi=300)
plt.title('An activity-selection problem', fontsize=14)
plt.plot(time, color='black', label='empyrical')

x = [0, 2000]
y = [0, min(time[-7:])]
plt.plot(x, y, color='b', label='theoretical', lw=5)

plt.ylabel('time, s.', fontsize=14)
plt.xlabel('n', fontsize=14)
plt.legend()
plt.show()