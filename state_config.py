#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:50:42 2024

@author: ariellefrommer
"""

## CODE FOR NEUTRINO CLASSES ##

import numpy as np
import re
import os

# class to load in states and state information
class StateConfig:
    def __init__(self, file_dir):
        self.file_dir = file_dir                                 # file directory
        self.states_path = os.path.join(file_dir, 'states.npy')  # states file
        self.notes_path = os.path.join(file_dir, 'config.dat')   # notes file
        self.states = None                                       # all activated states
        self.notes = []                                          # notes list
        self.init_state = None                                   # initial state of system
        self.j_to_bstr = {}                                      # index to binary string list
        self.bstr_to_j = {}                                      # binary string to index list
        self.Ns = None                                           # number of activated states
        self.Nn = None                                           # number of neutrinos
      #  self.activated_bins = []                                 # number of activated bins
        
        # functions to load and process state information
        self.load_files()
        self.process_notes()
        self.j_to_bstr, self.bstr_to_j = self.generate_lists()
        
    # loading in states and state information
    def load_files(self):
        self.states = np.load(self.states_path)
        with open(self.notes_path, 'r') as file:
            for note in file:
                note = note.strip()
                self.notes.append(note)    
    
    # processing state information
    def process_notes(self):
        
        # number of neutrinos
        self.neutrino_number = self.notes[0]
        self.Nn = int(re.search(r'\d+', self.neutrino_number).group())

        # initial state
        self.init_s = self.notes[1]
        self.init_state = ','.join(re.findall(r'\d+', self.init_s))
        
        # bins visited
        self.bins_visited = self.notes[2]
       # cleaned_str = re.sub(r'^.*?:\s*\[|\]$', '', self.bins_visited)
       # self.activated_bins = [int(num) for num in re.split(r',\s*', cleaned_str) if num]
        
    # generating list of states
    def generate_lists(self):
        for i in range(len(self.states)):
            oc = ','.join(str(x) for x in self.states[i])
            self.bstr_to_j[oc] = i
            self.j_to_bstr[i] = oc         
        self.Ns = len(self.j_to_bstr)
        return self.j_to_bstr, self.bstr_to_j
        
    # printing list of states
    def print_basis(self):
        print(f'For {self.Nn} neutrinos with an initial state of [{self.init_state}], {self.Ns} states are activated:')
        for i in range(self.Ns):
            print(f'{i}th state: ({self.j_to_bstr[i]})')
    
    # setting class information
    def __repr__(self):
        return f'{self.neutrino_number}\n{self.init_s}\n{self.bins_visited}'
    
    
        