# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 13:07:17 2023

@author: Timofei Chernobrovkin (412642) group J4133c
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

'''
###================================================================###
============================### Task I ###============================
###================================================================###
'''

###================================================================###
###==================== random adjacency matrix ===================###
###================================================================###

V = 100
E = 200
n = E
adj_mtrx = np.zeros((V, V))

while n != 0:
    i, j = np.random.randint(0, 99), np.random.randint(0, 99)
    if i != j and adj_mtrx[i, j] != 1:
        adj_mtrx[i, j], adj_mtrx[j, i] = 1, 1
        n -= 1

###================================================================###
###=========== matrix conversion to the adjacency list ============###
###================================================================###

adj_lst = {i: [] for i in range(V)}
for k, v in adj_lst.items():
    for i in range(V):
        adj_lst[k].append(i) if adj_mtrx[k, i] == 1 else next

###================================================================###
###====================== graph visualisation =====================###
###================================================================###

class GraphVisualization:
    def __init__(self, temp):
        self.visual = temp
    def visualize(self):
        G = nx.Graph()
        plt.figure(figsize=(10,7), dpi=300)
        plt.title(f'Graph with V = {V} and E = {E}', fontsize = 16)
        G.add_edges_from(self.visual)
        nx.draw_networkx(G)
        plt.show()

lst = []
for k, v in adj_lst.items():
    lst.extend([[k, i] for i in v])
G = GraphVisualization(lst)
G.visualize()

print('several rows of the matrix adjacency:')
print(adj_mtrx[42:46])
print('')
print('###=====================###')
print('')
print('several rows of the adjacency list:')
for i in range(8):
    print(i, adj_lst[i])

'''
###================================================================###
============================### Task II ###===========================
###================================================================###
'''

print('')
print('Task II')
print('')

###================================================================###
###======================= depth-first search =====================###
###================================================================###

print('')
print('depth-first search')
print('')

def DFS(temp, v, visited, adj_lst):
    visited[v] = True  
    temp.append(v)
    
    for i in adj_lst[v]:
        if visited[i] == False:
            temp = DFS(temp, i, visited, adj_lst)
    return temp

def connectedComponents(V, adj_lst):
    visited = []
    cc = []
    for i in range(V):
        visited.append(False)
    for v in range(V):
        if visited[v] == False:
            temp = []
            cc.append(DFS(temp, v, visited, adj_lst))
    return cc

cc = connectedComponents(V, adj_lst)

num_cc = 0
num_uncc = 0
for i in cc:
    if len(i) > 1:
        num_cc += len(i)
    else:
        num_uncc += 1

###================================================================###
###====================== breadth-first search ====================###
###================================================================###

print('')
print('breadth-first search')
print('')

def BFS(s, e, n, g):
    prev = solve(s, n, g)
    return reconstructPath(s, e, prev)

def solve(s, n, g):
    q = [] 
    q.insert(0, s) 
    visited = [False for i in range(n)]
    visited[s] = True
    
    prev = [-1 for i in range(n)]
    while len(q) != 0:
        node = q.pop()
        neighbours = g[node]
        
        for nei in neighbours:
            if visited[nei] == False:
                q.insert(0, nei)
                visited[nei] = True
                prev[nei] = node
    return prev

def reconstructPath(s, e, prev):
    path = []
    at = e
    while at != -1:
        path.append(at)
        at = prev[at]

    path.reverse()
    
    if path[0] == s:
        return path
    return []

se = np.random.randint(0, 99, size=2)
print('Start:', se[0])
print('finish:', se[1])

way = BFS(se[0], se[1], V, adj_lst)
print(f'The shortest way: {way}')
print(f'The founded way from {se[0]} to {se[1]}:')

for i in way:
    print(f"{i}: {adj_lst[i]}")