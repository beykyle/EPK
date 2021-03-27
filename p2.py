#! /usr/bin/python3

import numpy as np

from solver import *
import matplotlib.pyplot as plt
'''
Problem C.2 on 551 HW 4
'''

#
test()

# set up time grid
timesteps = 501
t = np.linspace(0,6.0001,num=timesteps)

# set up kinetics data
d = ConstantKineticsData()
d.mgt = 2.6E-15

# precursor data
oneg = False
if oneg:
    d.lambda_precursor  = np.array([0.49405, 0.0001, 0.00001,0.00001])
    d.beff              = np.array([0.76, 0.0001, 0.00001, 0.00001])
    d.beff = d.beff/100
else:
    d.lambda_precursor  = np.array([0.0128, 0.0318, 0.119, 0.3181, 1.4027, 3.9286])
    d.beff              = np.array([0.02584, 0.152, 0.13908, 0.30704, 0.1102, 0.02584])
    d.beff = d.beff/100

# thermal feedback dynamics data
d.lambda_H = 0#1.
d.gamma_D = 0#-1.2

# build data object gridded over time steps
data = Data.buildFromConstant(d, t)

# build a reactivity ramp
times = [0,1,6] # total asym ramp time: 6s
rho_ramp_up = LinearReactivityRamp(0,0.5 * 0.0076, 1) # 0$ -> 0.5$ in 1s
rho_ramp_down = LinearReactivityRamp(0.5 * 0.0076, 0, 5) #0.5$ -> 0$ in 5 s
rho = PieceWiseReactivityRamp(times , [rho_ramp_up, rho_ramp_down], t)

# set up plotter and solver, get analytic soln
solver = Solver(data,t,rho)
#power_analytic = solver.analyticPower1DG()
solver.solve(1, True)

plt.plot(solver.t, solver.p, "k.")
plt.show()


