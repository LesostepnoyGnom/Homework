# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:58:09 2023

@author: Timofei Chernobrovkin (412642) group J4133c
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time
import networkx as nx
import collections
import random

'''
###================================================================###
============================### Task I ###============================
###================================================================###
'''

n_V = 100
n_E = 500
n = n_E
adj_mtrx = np.zeros((n_V, n_V))

while n != 0:
    i, j = random.randint(0, 99), random.randint(0, 99)
    if i != j and adj_mtrx[i, j] == 0:
        weight = random.randint(0, 99)
        adj_mtrx[i, j], adj_mtrx[j, i] = weight, weight
        n -= 1

adj_lst = {i: [] for i in range(n_V)}
for k, v in adj_lst.items():
    for i in range(n_V):
        adj_lst[k].append([i, adj_mtrx[k, i]]) if adj_mtrx[k, i] != 0 else next
        
class My_Graph:
    def __init__(self, temp, vertices):
        self.V = vertices
        self.graph = temp
    
    def timer(func):
        def wrapper(*args, **kwargs):
            before = time.time()
            func(*args, **kwargs)
            time_check = time.time() - before
            print(f'\nThe algorithm took: {time_check} seconds')
            return time_check
        return wrapper
    
    def dist_print(self, dist, src):
        print(f'Vertex Distance from starting node {src}')
        for i in range(len(dist)):
            print(f'{i}\t\t{dist[i]}')
    
    def visualize(self):
        G = nx.Graph()
        plt.figure(figsize=(15,15), dpi=300)
        plt.title(f'Graph with {n_V} vertices and {n_E} edges',
                  fontsize=25)
        visualize = [i[0:2] for i in self.graph]
        G.add_edges_from(visualize)
        nx.draw_networkx(G)
        plt.show()

    @ timer
    def BF(self, src, show=True):
        dist = np.array([math.inf for i in range(self.V)])
        dist[src] = 0

        for _ in range(self.V - 1):
            for u, v, w in self.graph:
                if dist[u] != math.inf and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w

        for u, v, w in self.graph:
            if dist[u] != math.inf and dist[u] + w < dist[v]:
                print('There is a negative weight cycle in current graph')
                return
        
        if show == True:
            self.dist_print(dist, src)
    
    @ timer

    def Dijkstra(self, src, show=True):
        G = nx.Graph()
        G.add_nodes_from(list(set([i[0] for i in self.graph])))
        G.add_weighted_edges_from([i for i in self.graph])
        
        distances = nx.single_source_dijkstra(G, src)[0]
        
        sorted_distances = collections.OrderedDict(sorted(distances.items()))
        dist_list = [sorted_distances[i] for i in range(len(sorted_distances)) ]

        if show == True:
            self.dist_print(dist_list, src)

lst = []
for k, v in adj_lst.items():
    lst.extend([[k, i[0], i[1]] for i in v])
G = My_Graph(lst, n_V) 

G.visualize() 

time_lst_BF = []
time_lst_DJ = []

for i in range(10):
    np.random.seed(1)
    src = random.randint(0, 99)
    time_lst_BF.append(G.BF(src, show=False))  # Run BF algorithm
    time_lst_DJ.append(G.Dijkstra(src, show=False))  # Run Dijkstra's algorithm

print('\nAverage time required for BF algorithm =', round(sum(time_lst_BF)/len(time_lst_BF), 5), 'seconds')
print('\nAverage time required for Dijkstra algorithm =', round(sum(time_lst_DJ)/len(time_lst_DJ), 5), 'seconds')

'''
###================================================================###
============================### Task II ###============================
###================================================================###
'''

G = nx.grid_2d_graph(10, 20)

for edge in G.edges:
    G.edges[edge]['weight'] = 1

G.add_edges_from([((x, y), (x+1, y+1)) 
                  for x in range(9) 
                  for y in range(19)] + 
                 [((x+1, y), (x, y+1))
                  for x in range(9)
                  for y in range(19)])

plt.figure(figsize=(10,10), dpi=300)

pos = nx.spring_layout(G, iterations=1000, seed=42)
nx.draw(G, pos, node_size=40)
plt.show()

removed = []
while (len(removed) != 41):
    point = (random.randint(0,9), random.randint(0,19))
    if point not in removed:
        removed.append(point)

plt.figure(figsize=(10,10), dpi=300)

G.remove_nodes_from(removed)
print("Obstacle cells", removed)

nx.draw(G, pos, node_size=40)
plt.show()

def euclidean(start, target):
    h = math.sqrt((target[0] - start[0]) ** 2 + (target[1] - start[1]) ** 2)
    return h

def find_path(start, target):
    print(f'Find path from {start} to {target}')
    path = nx.astar_path(G, start, target, euclidean)
    print(f'Founded path: {path}\n')
    return path

all_paths = []
for i in range(5):
    while True:
        start = (random.randint(0, 9), random.randint(0, 19))
        print(start)
        target = (random.randint(0, 9), random.randint(0, 19))
        print(target)
        if (start not in removed) and (target not in removed):
            break
    all_paths.append(find_path(start, target))
    
def plot_paths(G, pos, all_paths):
    color_list = ['b', 'r', 'lime', 'm', 'orange']
    for path, color in zip(all_paths, color_list):
    
        plt.figure(figsize=(10,10), dpi=300)
        
        nx.draw(G, pos, node_size=20, node_color='k')
        nx.draw_networkx_nodes(G, 
                               pos, 
                               nodelist=path, 
                               node_color=color,
                               node_size=100)
        plt.show()

plot_paths(G, pos, all_paths)