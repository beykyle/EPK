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
d.beff              = np.array([0.0076])

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

# set up solver
solver = Solver(data,t,rho)

# set up plotters
power_plotter = Plotter(t)
react_plotter = Plotter(t)
precursor_plotter = Plotter(t, ylabel=r"$\zeta(t)$ [1/s]")

# compare numerical w/ analytic, theta study
solver.solve(0.5)
power_plotter.addData(solver.p, label=r"EPK, $\theta=\frac{1}{2}$", marker="k.")
precursor_plotter.addData(solver.zetas[0,:], label=r"EPK, $\theta=\frac{1}{2}$", marker="k.")
solver.reset()
solver.solve(1)
power_plotter.addData(solver.p, label=r"EPK, $\theta=1$", marker="y.", alpha=0.3)
precursor_plotter.addData(solver.zetas[0,:], label=r"EPK, $\theta=1$", marker="y.", alpha=0.3)
solver.reset()
#solver.debug = True
#solver.solve(0)
#power_plotter.addData(solver.p, label=r"EPK, $\theta=0$", marker=".", alpha=0.5)
#precursor_plotter.addData(solver.zetas[0,:], label=r"EPK, $\theta=0$", marker=".", alpha=0.5)

# plot analytic soln on top
power_analytic = solver.analyticPower1DG()
power_plotter.addData(power_analytic, label="Analytic", marker="--")

# plot reactivity
react_plotter.plotReactivityRamp(rho, data.beff)

# save figures
power_plotter.save("./results/theta_study.pdf")
react_plotter.save("./results/p1_rx.pdf")
precursor_plotter.save("./results/theta_study_zeta.pdf")

#
# time step study
timesteps = np.logspace(2,6, num=15)
l2_diff_CN = []
l2_diff_IMP = []
RDM_diff_CN = []
RDM_diff_IMP = []

def L2(a,b):
    return np.linalg.norm(a-b) / np.linalg.norm(a)

def relDiffMax(a,b):
    return (np.max(a) - np.max(b))/np.max(a)

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
    l2_diff_CN.append(L2(power_analytic,solver.p))
    RDM_diff_CN.append(relDiffMax(power_analytic,solver.p))
    solver.reset()
    solver.solve(1.0)
    l2_diff_IMP.append(L2(power_analytic,solver.p))
    RDM_diff_IMP.append(relDiffMax(power_analytic,solver.p))
    solver.solve(1.0)
    solver.reset()

p = Plotter(6.0E3/timesteps,  xlabel=r"time step [ms]", ylabel=r"$(L_2(p_{analytic} -  p_{epk}))/L_2(p_{analaytic})$")
p.addData(np.array(l2_diff_CN), marker="kx", label="Crank-Nicholson")
p.addData(np.array(l2_diff_IMP), marker="yx", label="Implicit", alpha=0.4)
p.addData(np.ones((len(RDM_diff_CN),)) * 0.01, label="0.01")
p.save("./results/time_step.pdf")

p = Plotter(6.0E3/timesteps,  xlabel=r"time step [ms]", ylabel=r"$(p^{max}_{analytic} -  p^{max}_{epk})/p^{max}_{analaytic}$")
p.addData(np.array(RDM_diff_CN), marker="kx", label="Crank-Nicholson")
p.addData(np.array(RDM_diff_IMP), marker="yx",label="Implicit", alpha=0.4)
p.addData(np.ones((len(RDM_diff_CN),)) * 0.01, label="0.01")
p.save("./results/time_step_RDM.pdf")
