#! /usr/bin/python3

import numpy as np

from solver import *

#
test()

# set up time grid
timesteps = 1001
t = np.linspace(0,6.0001,num=timesteps)

# set up kinetics data
precursor_groups = 1
d = ConstantKineticsData(precursor_groups)
d.beff = 0.76
d.mgt = 2.6E-15
d.lambda_precursor  = d.lambda_precursor * 0.49405

# thermal feedback dynamics data

# build data object gridded over time steps
data = Data.buildFromConstant(d, t)

# build a reactivity ramp
times = [0,1,6]
rho_ramp_up = LinearReactivityRamp(0,0.5 * d.beff, 1)
rho_ramp_down = LinearReactivityRamp(0.5 * d.beff, 0, 5)
rho = PieceWiseReactivityRamp(times , [rho_ramp_up, rho_ramp_down])

# run the solver
solver = Solver(data,t,rho)
solver.solve(0.5)
power_numeric = solver.p

# get the analytic solution
power_analytic = solver.analyticPower1DG()

# plot reactivity ramp
#plot_rx = Plotter(t)
#plot_rx.plotReactivityRamp(rho)
#plot_rx.save("./p1_rx.pdf")

# compare analytic and numeric power with the plotter
plotter = Plotter(t)
plotter.addData(power_analytic, "Analytic")
plotter.addData(power_numeric,  "EPKE Solver")
plotter.save("./results/p1.pdf")
