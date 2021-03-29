#! /usr/bin/python3

import numpy as np

from solver import *
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate as si
'''
Problem D.2 on 551 HW 4
'''

test()

#load in PARCS data
parcs_data = pd.read_csv("PARCS.dat", sep = " ")

def interp(qty, t):
    f = si.interp1d(parcs_data["Time"], parcs_data[qty])
    return f(t)

#set up time mesh
start_time, end_time = parcs_data["Time"].iloc[0], parcs_data["Time"].iloc[-1]
dt = 0.1e-3
t = np.arange(start_time, end_time + dt, dt)
#t = np.linspace(start_time, end_time, 10000)

#set up precursors
lambda_precursor = np.array([0.0128, 0.0318, 0.119, 0.3181, 1.4027, 3.9286])
beff = np.array([0.02584, 0.152, 0.13908, 0.30704, 0.1102, 0.02584])/100
lambda_precursor = np.average(lambda_precursor, weights = beff)
beff = beff.sum()

#create Data instance
d = ConstantKineticsData()
d.beff = beff
d.lambda_precursor = lambda_precursor
data = Data.buildFromConstant(d, t)
data.mgt = interp("Generation-Time", t)

#create Rho instance
rho = ReactivityGrid(t, interp("Reactivity", t))

#create the solver instance
solver = Solver(data, t, rho)
solver.solve(0.5)
plt.plot(solver.t, solver.p, "k.")
plt.show()
