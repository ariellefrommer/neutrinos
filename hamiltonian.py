#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:50:34 2024

@author: ariellefrommer
"""

import numpy as np

# class to calculate the Hamiltonian
class Hamiltonian:
    def __init__(self, momenta, config, dt):
        self.P = momenta.P                 # momentum lattice
        self.pairs = momenta.pairs         # momentum pairs
        self.K = momenta.K                 # number of momentum modes
        self.Ns = config.Ns                # number of states
        self.j_to_bstr = config.j_to_bstr  # index to binary string list
        self.bstr_to_j = config.bstr_to_j  # binary string to index list
        self.dt = dt                       # time step
        self.Nn = config.Nn                # number of neutrinos
        
        # creating Hamiltonian matrices as attributes
        self.H_full, self.U_full = self.construct_Hamiltonian()
        
    # state to binary conversion
    def b_to_j(self, b):
        oc = [i for i, bit in enumerate(b) if bit == 1]
        bstr = ','.join(str(x) for x in oc)
        return self.bstr_to_j[bstr]

    # binary to state conversion
    def j_to_b(self, j):
        bstr = self.j_to_bstr[j]
        oc = [int(x) for x in bstr.split(',')]
        b = [0] * self.K
        for i in range(len(oc)):
            b[oc[i]] = 1
        return b

    # applying a*(b1)a(b2) to a basis state
    def quad(self, b, basis):
        basis_copy = basis.copy()
        l = 1
        f = 1.0
        if basis_copy[b[1]] == 0:
            l = 0
        else:
            basis_copy[b[1]] = 0
            f *= (-1) ** np.sum(basis_copy[:b[1]])
        if basis_copy[b[0]] == 1:
            l = 0
        else:
            basis_copy[b[0]] = 1
            f *= (-1) ** np.sum(basis_copy[:b[0]])
        return l, f, basis_copy
    
    # applying a*(b1)a*(b2)a(b3)a(b4) to a basis state
    def quar(self, b, basis):
        basis_copy = basis.copy()
        l = 1
        f = 1.0
        if basis_copy[b[3]] == 0:
            l = 0
        else:
            basis_copy[b[3]] = 0
            f *= (-1) ** np.sum(basis_copy[:b[3]])
        if basis_copy[b[2]] == 0:
            l = 0
        else:
            basis_copy[b[2]] = 0
            f *= (-1) ** np.sum(basis_copy[:b[2]])
        if basis_copy[b[1]] == 1:
            l = 0
        else:
            basis_copy[b[1]] = 1
            f *= (-1) ** np.sum(basis_copy[:b[1]])
        if basis_copy[b[0]] == 1:
            l = 0
        else:
            basis_copy[b[0]] = 1
            f *= (-1) ** np.sum(basis_copy[:b[0]])
        return l, f, basis_copy

    # defining kinetic energy term applied to basis state j
    def kinetic(self, j):
        sin = self.j_to_b(j)
        state = np.zeros(self.Ns) * 1j

        # applying operators for all momentum states
        for p in range(self.K):
            absp = np.sqrt(np.sum(self.P[p] * self.P[p]))
            t, fa, sout = self.quad([p, p], sin)
            if t == 1:
                state[self.b_to_j(sout)] += fa * absp
        return state

    # defining spherical function for g factor calculation
    def p_spherical(self, p):
        absp = np.sqrt(np.sum(p*p))
        theta = np.arctan2(np.sqrt(p[0]**2+p[1]**2), p[2])
        phi = np.arctan2(p[1], p[0])
        return absp, theta, phi
   
    # calculating g factor for interaction potential
    def g_factor(self, p1, p2, q1, q2):
            absp1, tp1, pp1 = self.p_spherical(p1)
            absp2, tp2, pp2 = self.p_spherical(p2)
            absq1, tq1, pq1 = self.p_spherical(q1)
            absq2, tq2, pq2 = self.p_spherical(q2)
            fac1 = np.exp(-1j * pq1) * np.sin(tq1 / 2.) * np.cos(tq2 / 2.) - np.exp(-1j * pq2) * np.cos(tq1 / 2.) * np.sin(tq2 / 2.)
            fac2 = np.exp(1j * pp1) * np.sin(tp1 / 2.) * np.cos(tp2 / 2.) - np.exp(1j * pp2) * np.cos(tp1 / 2.) * np.sin(tp2 / 2.)
            return 2 * fac1 * fac2
          
          #  return 1
    
     # defining the interaction potential function
    def interaction_potential(self, j):
        sin = self.j_to_b(j)
        state = np.zeros(self.Ns) * 1j
        for i in range(len(self.pairs)):
            p1, p2, q1, q2 = self.pairs[i]
            factor = - self.g_factor(self.P[p1], self.P[p2], self.P[q1], self.P[q2])
            t, fa, sout = self.quar([p1, p2, q1, q2], sin)
            if t == 1:
                state[self.b_to_j(sout)] += fa * factor
        return state
    
    # constructing Hamiltonian
    def construct_Hamiltonian(self):
        H_vac = np.zeros((self.Ns, self.Ns)) * 1j
        H_vv_full = np.zeros((self.Ns, self.Ns)) * 1j

        # calculating kinetic energy and interaction potential
        for i in range(self.Ns):
            H_vac[:, i] = self.kinetic(i)
            if i % 50 == 0:
                print(f'Constructing {i}th column of the Hamiltonian')
            H_vv_full[:, i] = self.interaction_potential(i)

        # creating Hamiltonian matrix
        H_full = H_vac + H_vv_full
        hvals_full, hvecs_full = np.linalg.eigh(H_full) # print eigenvalues
        U_full = hvecs_full @ np.diag(np.exp(-self.dt * hvals_full * 1j)) @ hvecs_full.conj().T
        return H_full, U_full
    