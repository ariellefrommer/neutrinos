#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:55:35 2024

@author: ariellefrommer
"""
import numpy as np

# class to make momentum pairs
class Momenta:
    def __init__(self, P):
        self.P = P                        # momentum matrix
        self.K = len(P)                   # number of momentum modes
        self.pairs = self.create_pairs()  # creating momentum pairs
        
    # defining momentum conservation function
    def p_is_equal(self, p1, p2):
        diff = p1 - p2
        return np.isclose(np.sum(np.abs(diff)), 0)

    # defining kinetic energy conservation function
    def ke_is_equal(self, p1, p2, p3, p4):
        
        # getting magnitudes of momenta
        k1 = np.linalg.norm(p1) 
        k2 = np.linalg.norm(p2) 
        k3 = np.linalg.norm(p3) 
        k4 = np.linalg.norm(p4)
        return np.isclose(k1 + k2 - k3 - k4, 0)
    
    # creating momentum pairs
    def create_pairs(self):
        P = self.P
        momenta = []
        
        # looping through momentum lattice
        for i1 in range(self.K):
            for i2 in range(self.K):
                for i3 in range(self.K):
                    for i4 in range(self.K):
                        
                        # momentum conservation
                        if self.p_is_equal(P[i1] + P[i2], P[i3] + P[i4]):
                            
                            # kinetic energy conservation
                            if self.ke_is_equal(P[i1], P[i2], P[i3], P[i4]):
                                momenta.append([i1, i2, i3, i4])
        
        # grabbing all momentum pairs
        return np.array(momenta)
    
    # setting momentum class information
    def __repr__(self):
        return f'Total number of conserved momenta pairs for {self.K} momentum modes: {len(self.pairs)}'
    
    
    