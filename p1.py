#! /usr/bin/python3

import numpy as np

from solver import *
import matplotlib.pyplot as plt
'''
Problem C.1 on 551 HW 4
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
d.lambda_precursor  = np.array([0.49405])
d.beff              = np.array([0.76])

# thermal feedback dynamics data
d.lambda_H = 0
d.gamma_D = 0

# build data object gridded over time steps
data = Data.buildFromConstant(d, t)

# build a reactivity ramp
times = [0,1,6] # total asym ramp time: 6s
rho_ramp_up = LinearReactivityRamp(0,0.5 * d.beff, 1) # 0$ -> 0.5$ in 1s
rho_ramp_down = LinearReactivityRamp(0.5 * d.beff, 0, 5) #0.5$ -> 0$ in 5 s
rho = PieceWiseReactivityRamp(times , [rho_ramp_up, rho_ramp_down], t)

# set up plotter and solver, get analytic soln
solver = Solver(data,t,rho)
power_analytic = solver.analyticPower1DG()
power_plotter = Plotter(t)

# compare numerical w/ analytic, theta study
solver.solve(0.5)
power_plotter.addData(solver.p, label=r"EPK, $\theta=\frac{1}{2}$", marker=".")
solver.solve(1)
power_plotter.addData(solver.p, label=r"EPK, $\theta=1$", marker=".", alpha=0.5)
#solver.debug = True
solver.solve(0)
power_plotter.addData(solver.p, label=r"EPK, $\theta=0$", marker=".", alpha=0.5)

# plot analytic soln on top
power_plotter.addData(power_analytic, label="Analytic", marker="--")

# plot reactivity
react_plotter = Plotter(t)
react_plotter.plotReactivityRamp(rho, data.beff)

# save figures
power_plotter.save("./results/theta_study_nofeedback.pdf")
react_plotter.save("./results/p1_rx.pdf")

# time step study
timesteps = [101, 201, 501, 1001, 2001, 5001, 10001]
l2_diff = []
for num_t in timesteps:
    # make new time grid
    t = np.linspace(0,6.0001,num=num_t)
    # set up data and reactivity on this time grid
    data = Data.buildFromConstant(d, t)
    rho = PieceWiseReactivityRamp(times , [rho_ramp_up, rho_ramp_down], t)
    # set up and run solver
    solver = Solver(data,t,rho)
    power_analytic = solver.analyticPower1DG()
    solver.solve(0.5)
    l2_diff.append(np.linalg.norm(power_analytic - solver.p))


p = Plotter(6.0/np.array(timesteps),  xlabel=r"time step [s]", ylabel=r"$L_2(p_{analytic}, p_{epk})$")
p.addData(np.array(l2_diff))
p.save("./results/time_step.pdf")
