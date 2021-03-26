#! /usr/bin/python3

import numpy as np

from solver import *
import matplotlib.pyplot as plt

#
test()

# set up time grid
timesteps = 151
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

# run the solver
solver = Solver(data,t,rho, debug=True)
solver.solve(0.5)

power_numeric = solver.p

# get the analytic solution
power_analytic = solver.analyticPower1DG()

# plot reactivity ramp
plot_rx = Plotter(t, ylabel=r"$\rho$ [\$]")
plot_rx.plotReactivityRamp(rho, data.beff)
plot_rx.save("./results/p1_rx.pdf")

# compare analytic and numeric power with the plotter
plotter = Plotter(t)
plotter.addData(power_analytic, label="Analytic", marker="x")
plotter.addData(power_numeric,  label="EPKE Solver", marker="--")
#plt.show()
plotter.save("./results/p1.pdf")
