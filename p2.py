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
lambda_precursor  = np.array([0.0128, 0.0318, 0.119, 0.3181, 1.4027, 3.9286])
beff              = np.array([0.02584, 0.152, 0.13908, 0.30704, 0.1102, 0.02584])/100

# build a reactivity ramp
times = [0,1,6] # total asym ramp time: 6s
rho_ramp_up = LinearReactivityRamp(0,0.5 * beff.sum(), 1) # 0$ -> 0.5$ in 1s
rho_ramp_down = LinearReactivityRamp(0.5 * beff.sum(), 0, 5) #0.5$ -> 0$ in 5 s
rho = PieceWiseReactivityRamp(times , [rho_ramp_up, rho_ramp_down], t)

# beta weighted precursor data
lambda_precursor_sets = {}

lambda_precursor_sets["beta"] = np.average(lambda_precursor, weights = beff)
lambda_precursor_sets["inv"] = beff.sum() / np.sum(beff / lambda_precursor)
lambda_precursor_sets["6-group"] = lambda_precursor.copy()

for key, value in lambda_precursor_sets.items():
    d.lambda_precursor = value
    if key in ["inv", "beta"]:
        d.beff = beff.sum()
    else:
        d.beff = beff
    data = Data.buildFromConstant(d, t)
    solver = Solver(data, t, rho)
    solver.solve(0.5, False)
    plt.plot(solver.t, solver.p, label = key)

d.lambda_precursor = lambda_precursor_sets["6-group"]
d.beff = beff
data = Data.buildFromConstant(d, t)
solver = Solver(data, t, rho, time_dep_precurs = True)
solver.solve(0.5, False)
plt.plot(solver.t, solver.p, label = "time-dep")
plt.ylim([0.5, 4])
plt.legend()
#plt.show()


d = Data.buildFromConstant(d,t)
beta_weighted_data = d.collapseGroups(beta_weighted=True)
invbeta_weighted_data  = d.collapseGroups(beta_weighted=False)

p = Plotter(t)
s = Solver(d,t,rho)
s.solve(0.5)
p.addData(s.p, label="6G")
s.reset(d=beta_weighted_data)
s.solve(0.5)
p.addData(s.p, label=r"1G, $\beta$-weighted")
s.reset(d=invbeta_weighted_data)
s.solve(0.5)
p.addData(s.p, label=r"1G, $\frac{1}{\beta}$-weighted")
p.save("6G_asym_ramp.pdf")
