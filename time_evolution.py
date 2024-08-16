#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:59:06 2024

@author: ariellefrommer
"""

import numpy as np
import matplotlib.pyplot as plt
import os
 
# class to time-evolve the system and plot observables
class TimeEvolution:
    def __init__(self, init_state, Nt, hamiltonian):
        self.init_state = init_state            # initial state
        self.Nt = Nt                            # number of time steps
        self.bstr_to_j = hamiltonian.bstr_to_j  # binary string to index list
        self.Ns = hamiltonian.Ns                # number of states
        self.dt = hamiltonian.dt                # time step
        self.K = hamiltonian.K                  # number of bins
        self.Nn = hamiltonian.Nn                # number of neutrinos
        self.U_full = hamiltonian.U_full        # Hamiltonian matrix     
        self.j_to_b = hamiltonian.j_to_b        # binary to state conversion
        
        # creating probability amplitudes, number operators, and time as attributes 
        self.c_full, self.obs_store_full = self.time_evolve()               
        self.time = self.obs_store_full[:, self.K] 
        self.eq_values = self.get_eq_values()
        self.eq_diff = self.get_eq_diff()
        
    # norm and observables of interest
    def norm(self, state):
        return np.sqrt(np.sum(state * state.conj()))

    # input is the state vector, returns the # of neutrino in each bin
    def observable(self, state):
        obs = np.zeros(self.K)
        for i in range(self.Ns):
            binary = self.j_to_b(i)
            obs += np.abs(state[i])**2 * np.array(binary)
        return obs
    
    # time-evolving the system
    def time_evolve(self):
        
        # preparing functions to store amplitudes and number operators
        c_full = np.zeros((self.Nt, self.Ns))
        obs_store_full = np.zeros((self.Nt, self.K + 1))

        # creating array for storing the state, and choose the initial state
        initbin = self.init_state
        initj = self.bstr_to_j[initbin]
        state_full = np.zeros(self.Ns) * 1j
        state_full[initj] = 1.0 

        # calculating probability amplitudes
        for i in range(self.Nt):
            state_full = self.U_full @ state_full
            n_full = self.norm(state_full)
            if abs(n_full - 1.) > 1e-5:
                print('Norm off by > 1e-5 at time ', i * self.dt)
                break

            # storing amplitudes and time
            obs_full = self.observable(state_full)
            obs_store_full[i, :self.K] = obs_full
            obs_store_full[i, self.K] = i * self.dt
            c_full[i, :] = np.abs(state_full)**2
        return c_full, obs_store_full
        
    # plotting Hamiltonian
    def plot_Hamiltonian(self, file_path):
        
        # creating figure
        fig = plt.figure(figsize = (12, 8))
        gs = fig.add_gridspec(1, 1)
        axs = gs.subplots()
        
        fig.patch.set_facecolor('white')
        axs.set_facecolor('white')
        
        # indices to be plotted
        states = range(self.Ns)
        
        # labeling states
        labels = [f'{states[i]}th state' for i in states]
            
        # plotting Hamiltonian
        for idx, label in enumerate(labels):
            axs.plot(self.time, self.c_full[:, states[idx]], label = label)

        # plotting sum of specified states for Full Hamiltonian
        axs.plot(self.time, np.sum(self.c_full[:, states], axis=1), label='Sum')
       
        # setting subplot parameters
        axs.set_title(f'Full Hamiltonian with {self.Nn} Neutrinos and {self.Ns} States', fontsize = 14)
        axs.set_xlabel('time ($\epsilon$)', fontsize = 14)
        axs.set_ylabel('Amplitude', fontsize = 14)
        
        # setting global parameters
        plt.tick_params(axis='both', which='major', labelsize = 12)
        legend = plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0125), ncol = 4,  frameon=True)
        legend.get_frame().set_edgecolor('black')
        
        # saving the plot
        file_path = os.path.join(file_path, 'Hamiltonian.png')
        plt.savefig(file_path, bbox_inches='tight', dpi = 200)
        
        # displaying and closing the plot
        plt.show()
        plt.close()
    
    # plotting number operators
    def plot_number_operators(self, file_path):
                
        # creating figure
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(1, 1)
        axs = gs.subplots()

        fig.patch.set_facecolor('white')
        axs.set_facecolor('white')

        # choosing a colormap
        colormap = plt.get_cmap('tab20')

        # indices to be plotted
        bins = range(self.K)

        # labeling bins
        labels = [f'Bin {bins[i]}' for i in bins]
        
        # generating colors from the colormap
        colors = [colormap(i / len(bins)) for i in bins]
        
        # plotting number operators
        for idx, color in zip(bins, colors):
            label = labels[idx]
            axs.plot(self.time, self.obs_store_full[:, idx], label = label, color = color)

        # setting subplot parameters
        axs.set_title(f'Number Operators with {self.Nn} Neutrinos and {self.Ns} States', fontsize=16)
        axs.set_xlabel('time ($\epsilon$)', fontsize=14)
        axs.set_ylabel('$N_i$', fontsize=14)
    
        # setting global parameters
        plt.tick_params(axis='both', which='major', labelsize=14)
        legend = plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0125), ncol=2, frameon=True)
        legend.get_frame().set_edgecolor('black')  
        
        # saving the plot
        file_path = os.path.join(file_path, 'number_operators.png')
        plt.savefig(file_path, bbox_inches = 'tight', dpi = 200)
        
        # displaying and closing the plot
        plt.show()
        plt.close()
       
    # getting equilibrium values
    def get_eq_values(self):
        eq_full = np.ones(self.Ns) / np.sqrt(self.Ns)
        eq = self.observable(eq_full)
        return eq
    
    # plotting equilibrium values
    def plot_eq(self, file_path):
        
        # choosing a colormap
        colormap = plt.get_cmap('tab20')

        # indices to be plotted
        bins = range(self.K)
        
        # labeling bins
        labels = [f'Bin {bins[i]}' for i in bins]   
                
        # generating colors from the colormap
        colors = [colormap(i / len(bins)) for i in bins]
          
        # calculating the number of subplots
        n_plots = len(bins)
        n_cols = 5
        n_rows = (n_plots + n_cols - 1) // n_cols 

        # creating figure and subplots
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))  # Adjust height as needed
        fig.patch.set_facecolor('white')


        # flatting axs for easy indexing, handle the case with only one row or one column
        if n_rows == 1:
            axs = axs[np.newaxis, :]
        if n_cols == 1:
            axs = axs[:, np.newaxis]
        axs = axs.flatten()

        print(bins)
        # plotting number operators in individual subplots
        for idx in bins:
            label = labels[idx]
            color = colors[idx]
            ax = axs[idx]
            ax.plot(self.time, self.obs_store_full[:, idx], label=label, color=color)
            ax.axhline(y=self.eq_values[idx], label=f'{label} eq.', color=color, linestyle='--', linewidth=2)
            ax.set_xlabel('time ($\epsilon$)', fontsize=12)
            ax.set_ylabel('$N_i$', fontsize=12)
            ax.legend(loc='upper left', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.set_facecolor('white')
            
        # hiding unused subplots
        for ax in axs[n_plots:]:
             ax.axis('off')
            
        # saving the plot
        file_path = os.path.join(file_path, 'equilibrium_values.png')
        plt.savefig(file_path, bbox_inches = 'tight', dpi = 200)
        
        # displaying the plo
        plt.tight_layout()
        plt.show()
        plt.close()
    
    def get_eq_diff(self):
        eq_diff = self.obs_store_full[:, :self.K] - self.eq_values
        return eq_diff
    
    # plotting difference
    def plot_eq_diff(self, file_path):
                
        # creating figure
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(1, 1)
        axs = gs.subplots()

        fig.patch.set_facecolor('white')
        axs.set_facecolor('white')

        # choosing a colormap
        colormap = plt.get_cmap('tab20')

        # indices to be plotted
        bins = range(self.K)

        # labeling bins
        labels = [f'Bin {bins[i]}' for i in bins]
        
        # generating colors from the colormap
        colors = [colormap(i / len(bins)) for i in bins]
        
        # plotting number operators
        for idx, color in zip(bins, colors):
            label = labels[idx]
            axs.plot(self.time, self.eq_diff[:, idx], label = label, color = color)

        # setting subplot parameters
        axs.set_title(f'Equilibration with {self.Nn} Neutrinos and {self.Ns} States', fontsize=16)
        axs.set_xlabel('time ($\epsilon$)', fontsize=14)
       # axs.set_ylabel('$N_i$', fontsize=14)
    
        # setting global parameters
        plt.tick_params(axis='both', which='major', labelsize=14)
        legend = plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0125), ncol=2, frameon=True)
        legend.get_frame().set_edgecolor('black')  
        
        # saving the plot
        file_path = os.path.join(file_path, 'equilibration.png')
        plt.savefig(file_path, bbox_inches = 'tight', dpi = 200)
        
        # displaying and closing the plot
        plt.show()
        plt.close()
        
        
        
    