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

# compare feedback numerical w/ analytic
solver = Solver(data,t,rho, debug=False)
solver.solve(0.5,feedback=True)
power_feed_solver = solver.p
power_analytic = solver.analyticPower1DG()


# plot analytic comparison
p = Plotter(t)
p.addData(power_analytic, label="Analytic", marker="--")
p.addData(power_analytic, label="EPK", marker="x")
p.save("./results/p1.pdf")

p = Plotter(t, ylabel=r"$\rho$ [\$]")
p.plotReactivityRamp(rho,data.beff)
p.save("./results/p1_rx.pdf")
