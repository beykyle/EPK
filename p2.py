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
d.mgt = 2.6E-5

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
lambda_precursor_sets["inverse"] = beff.sum() / np.sum(beff / lambda_precursor)
lambda_precursor_sets["6-group"] = lambda_precursor.copy()
lambda_precursor_sets["tmax"] = np.asarray(0.2439)

pa = Plotter(t) #power plot of all methos
pb = Plotter(t) #power plot of beta vs 6g
zb = Plotter(t, ylabel=r"$\zeta$") #zeta plot of 1g vs 6g

for key, value in lambda_precursor_sets.items():
    d.lambda_precursor = value
    print(value)
    if key in ["inverse", "beta", "tmax"]:
        l = key
        d.beff = beff.sum()
    else:
        l = ""
        d.beff = beff
    data = Data.buildFromConstant(d, t)
    solver = Solver(data, t, rho)
    solver.solve(0.5, False)
    pa.addData(solver.p, label = str(d.beff.size) + "G " +  l)
    if key in ["beta", "6-group"]:
        pb.addData(solver.p, label = str(d.beff.size) + "G " +  l)
        zb.addData(solver.zetas.sum(0), label = str(d.beff.size) + "G " +  l)


d.lambda_precursor = lambda_precursor_sets["6-group"]
d.beff = beff
data = Data.buildFromConstant(d, t)
solver = Solver(data, t, rho, time_dep_precurs = True)
solver.solve(0.5, False)
pc = Plotter(t[1:], ylabel = r"$\lambda$")
pc.addData(solver.d.lambda_precursor[0,1:])

pa.addData(solver.p, label = "1G time-dependent")
pa.save("./results/precursor_grouping_comparison.pdf")
pb.save("./results/1v6precurs_comparison.pdf")
zb.save("./results/1v6precurs_zeta_comparison.pdf")
pc.save("./results/tdep_lambda.pdf")
#plt.show()
exit()


d = Data.buildFromConstant(d,t)
beta_weighted_data = d.collapseGroups(beta_weighted=True)
invbeta_weighted_data  = d.collapseGroups(beta_weighted=False)

p = Plotter(t)
s = Solver(d,t,rho, debug=True)
print(s.p)
s.debug = False
p.addData(s.p, label="6G")
s.resetNewData(beta_weighted_data, t)
s.solve(0.5)
print(s.p)
p.addData(s.p, label=r"1G, $\beta$-weighted")
s.resetNewData(invbeta_weighted_data, t)
s.solve(0.5)
print(s.p)
p.addData(s.p, label=r"1G, $\frac{1}{\beta}$-weighted")
p.save("./results/6G_asym_ramp.pdf")
