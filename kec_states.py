#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 13:57:29 2024

@author: ariellefrommer
"""

import numpy as np
from itertools import combinations
import os

# making a momentum grid
dp = 1.0
pmax = 4 # changed from 5
pmin = -pmax
np1 = round((pmax-pmin)/dp) + 1

ps_grid = np.zeros((np1**2,3 ))
pxs, pys = np.meshgrid(np.linspace(pmin,pmax,np1), np.linspace(pmin,pmax,np1))
ps_grid[:,0] = pxs.reshape(np1**2)
ps_grid[:,1] = pys.reshape(np1**2)
ps_grid[:,2] = np.zeros(np1**2)

kes_grid = np.zeros(np1**2)
kes_grid = np.sqrt(ps_grid[:,0]**2+ps_grid[:,1]**2+ps_grid[:,2]**2)

ps = []
kes = []

# Impose ke <= pmax, ke > 0, and p_x > 0
for i in range(len(kes_grid)):
    if kes_grid[i] < pmax+1e-5 and kes_grid[i] > 0 and ps_grid[i,0] >  0:
        ps.append(ps_grid[i])
        kes.append(kes_grid[i])

np2 = len(kes)
ps = np.array(ps)
kes = np.array(kes)

np.save('/Users/ariellefrommer/Desktop/Neutrinos/KE_configurations/P_new', ps)

kec_nt = []
kec_condition = 1e-10

for i1 in range(np2):
    p1 = ps[i1,:]
    for i2 in range(i1,np2):
        p2 = ps[i2,:]
        for i3 in range(np2):
            p3 = ps[i3,:]
            for i4 in range(i3,np2):
                p4 = ps[i4,:]
                pdiff = p1 + p2 - p3 - p4
                if np.sum(np.abs(pdiff)) < 1e-10:
                    
                    # finding kinetic energy (magnitude of momentum)
                    k1 = np.sqrt(np.dot(p1,p1))
                    k2 = np.sqrt(np.dot(p2,p2))
                    k3 = np.sqrt(np.dot(p3,p3))
                    k4 = np.sqrt(np.dot(p4,p4))
                    if np.abs(k1 + k2 - k3 - k4) < kec_condition:
                        if i1 != i3:
                            kec_nt.append([i1,i2,i3,i4])
kec_nt = np.array(kec_nt)
Nkec = len(kec_nt)

class StateSearcher:
    def __init__(self, ps, kec_nt, ninit):
        self.ps = ps                                             # momentum matrix
        self.kec_nt = kec_nt                                     # kinetic energy conserving pairs
        self.ninit = ninit                                       # initial state
        self.Nn = len(self.ninit)                                # number of neutrinos
        self.pinit = np.sum(ps[ninit], axis=0)                   # initial total momentum
        self.keinit = np.sum(np.linalg.norm(ps[ninit], axis=1))  # initial kinetic energy
        
        # combinations of momentum modes
        self.binom = np.array(list(combinations(range(self.Nn), 2)))
        self.Nbinom = len(self.binom)
        
        # initial state setup
        self.p_states = np.array([ninit])
        self.newstate1 = np.array([np.zeros(self.Nn), ninit])
        self.trial = 0
        
        # final property
        self.activated_bins = list(set(self.p_states.flatten()))
    
    # printing initial configurations
    def print_init(self):
        print(f'Number of neutrinos is {self.Nn}')
        print(f'Reference state is {self.ninit}')
        print(f'Reference state total momentum: {self.pinit}')
        print(f'Reference state kinetic energy: {self.keinit}')
              
    # checking if the same momentum mode is used only once
    def check(self, state):
        return not np.any(np.diff(state) < 1e-7)

    def apply(self, state):
        newstate = []
        for i in range(self.Nbinom):
            k = np.array([state[self.binom[i, 0]], state[self.binom[i, 1]]])
            for j in range(len(self.kec_nt)):
                if np.all(np.isclose(k, self.kec_nt[j, :2])):
                    state_i = state.copy()
                    state_i[self.binom[i, 0]] = self.kec_nt[j, 2]
                    state_i[self.binom[i, 1]] = self.kec_nt[j, 3]
                    state_i = np.sort(state_i)
                    if self.check(state_i):
                        newstate.append(state_i)
        return np.array(newstate)

    # searching for new configurations
    def search_states(self, nnewstate = 10):
        while nnewstate > 1:
            newstate2 = np.zeros((1, self.Nn))
            for i in range(1, len(self.newstate1)):
                newi = self.apply(self.newstate1[i])
                if len(newi) > 0:
                    newstate2 = np.append(newstate2, newi, axis=0)
            
            self.newstate1 = np.array([np.zeros(self.Nn)])
            for j in range(1, len(newstate2)):
                dist = np.sum(np.abs(self.p_states - newstate2[j]), axis=1)
                if np.min(dist) > 1e-10:
                    self.p_states = np.append(self.p_states, [newstate2[j].astype(int)], axis=0)
                    self.newstate1 = np.append(self.newstate1, [newstate2[j]], axis=0)
            nnewstate = len(self.newstate1)    
        
        # printing all states that can be visited
        print(f'All {len(self.p_states)} states that can be visited from the reference state')
        for i in range(len(self.p_states)):
            print(f'{i}th: ', self.p_states[i])
        
    def save_config(self, file_path):
        
        config_path = os.path.join(file_path, 'config.dat')
        npy_path = os.path.join(file_path, 'states.npy')
    
        # preparing configuration data
        lines = [f'Number of neutrinos: {self.Nn}',
                 f'Initial state: {self.ninit}',
                 f'Activated bins: {self.activated_bins}']

        with open(config_path, 'w') as file:
            for line in lines:
                file.write(line + '\n')

        # saving numpy array to a .npy file
        np.save(npy_path, self.p_states)
        

