#! /usr/bin/python3

import numpy as np

from solver import *
import matplotlib.pyplot as plt

'''
A study on on how p(t) is impacted by linear feedback parameters
'''

#
test()

# set up time grid
timesteps = 4200
t = np.linspace(0,6.0001,num=timesteps)

# set up kinetics data
d = ConstantKineticsData()
d.mgt = 2.6E-15

# precursor data
d.lambda_precursor  = np.array([0.49405])
d.beff              = np.array([0.76])

# thermal feedback dynamics data

# build data object gridded over time steps
data = Data.buildFromConstant(d, t)

# build a reactivity ramp
times = [0,1,6] # total asym ramp time: 6s
rho_ramp_up = LinearReactivityRamp(0,0.5 * d.beff, 1) # 0$ -> 0.5$ in 1s
rho_ramp_down = LinearReactivityRamp(0.5 * d.beff, 0, 5) #0.5$ -> 0$ in 5 s
rho = PieceWiseReactivityRamp(times , [rho_ramp_up, rho_ramp_down], t)

#compare feedback
p = Plotter(t)

fig, ax = plt.subplots(1,1)
cs = ["k", "b", "r", "purple", "g"]
for i, gamma_D in enumerate([0, .3, 1., 2., 6]):
    solver = Solver(data, t, rho)
    solver.d.lambda_H += 1
    solver.d.gamma_D += -gamma_D
    solver.solve(0.5, feedback=True)
    p.addData(solver.p, label=r"$\gamma_d$ = %.1f"%gamma_D, marker=cs[i])

p.save("./results/gamma_study.pdf")
