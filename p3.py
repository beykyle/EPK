#! /usr/bin/python3

import numpy as np

from solver import *
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate as si
'''
Problem D.1 on 551 HW 4
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

t = t[:t.size//10]
t = parcs_data["Time"].values
print(t.shape)

#creating a better tmest
N = 20000
t = np.zeros(N)
t[:1000] = parcs_data["Time"].values[:1000]
t[1000:] = np.linspace(t.max()+dt, end_time, N - 1000)

#t = np.linspace(start_time, end_time, 1000)

#set up precursors
lambda_precursor = np.array([0.0128, 0.0318, 0.119, 0.3181, 1.4027, 3.9286])
beff = np.array([0.02584, 0.152, 0.13908, 0.30704, 0.1102, 0.02584])/100
#lambda_precursor = np.average(lambda_precursor, weights = beff)
#beff = beff.sum()

#create Data instance
d = ConstantKineticsData()
d.beff = beff
d.lambda_precursor = lambda_precursor
data = Data.buildFromConstant(d, t)
data.mgt = interp("Generation-Time", t)
data.f_fp = interp("Normalization-Factor", t)

#create Rho instance
rho = ReactivityGrid(t, interp("Reactivity", t))
rho.rho = rho.rho *np.sum(beff) # convert form $ to reactivity

#create the solver instance
power_plt = Plotter(t)
power_plt.addData(interp("Relative-Power",t)*1e6 , label="PARCS", logx = True)
solver = Solver(data, t, rho, debug=False)
solver.solve(0.5, feedback=True)
power_plt.addData(solver.p*interp("Normalization-Factor", t), label="EPK", logx = True)
power_plt.save("./results/parcs_d1.pdf")

plt.legend()
