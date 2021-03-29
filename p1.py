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
max_time = 6.00
timesteps = 100
t = np.linspace(0,max_time,num=timesteps)

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
tmax = 1.0
times = [0,tmax,max_time] # total asym ramp time: 6`0s
rho_ramp_up = LinearReactivityRamp(0,0.5 * d.beff[0], tmax) # 0$ -> 0.5$ in 1s
rho_ramp_down = LinearReactivityRamp(0.5 * d.beff[0], 0  , max_time - tmax)
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
solver.solve(0)
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
step_size = np.array([1E-3, 10E-3, 20E-3, 25E-3, 40E-3, 50E-3])
timesteps = np.round(max_time/step_size).astype(dtype=int) + 1
print(step_size)
print(timesteps)

l2_diff_CN = []
l2_diff_IMP = []
l2_diff_EXP = []
RDM_diff_CN = []
RDM_diff_IMP = []
RDM_diff_EXP = []

def L2(a,b):
    return np.linalg.norm(a-b) / np.linalg.norm(a)

def relDiffMax(a,b):
    return np.fabs((np.max(a) - np.max(b))/np.max(a))

for num_t in timesteps:
    # make new time grid
    t = np.linspace(0,max_time,num=num_t)
    p = Plotter(t)
    # set up data and reactivity on this time grid
    data = Data.buildFromConstant(d, t)
    rho = PieceWiseReactivityRamp(times , [rho_ramp_up, rho_ramp_down], t)
    # set up and run solver
    # get analytic
    solver = Solver(data,t,rho)
    power_analytic = solver.analyticPower1DG()
    max_anal = np.max(power_analytic)
    idx_max = np.argmax(power_analytic)
    # CN
    solver.solve(0.5)
    max_CN = solver.p[idx_max]
    p.addData(np.copy(solver.p), label="CN", marker="x")
    l2_diff_CN.append(L2(power_analytic,solver.p))
    RDM_diff_CN.append( np.fabs((max_anal - max_CN)/max_anal) )
    solver.reset()
    # implicit
    solver.solve(1.0)
    max_IM = solver.p[idx_max]
    l2_diff_IMP.append(L2(power_analytic,solver.p))
    RDM_diff_IMP.append( np.fabs((max_anal - max_IM)/max_anal) )
    p.addData(np.copy(solver.p), label="Implicit", marker="x", alpha=0.2)
    solver.reset()
    p.addData(power_analytic, label="analytic")
    p.save("./results/time_step_"+ str(num_t) + "_theta.pdf")
    # explicit
    #solver.solve(0.0)
    #l2_diff_EXP.append(L2(power_analytic,solver.p))
    #RDM_diff_EXP.append(relDiffMax(power_analytic,solver.p))
    #solver.reset()
    print("Time Step Size [ms]        : {:1.5f}".format(solver.dt[0]*1E3))
    print("Relative Max Diff (CN) [%] : {:1.5f}".format(RDM_diff_CN[-1]))
    print("Relative Max Diff (IM) [%] : {:1.5f}".format(RDM_diff_IMP[-1]))
    print("Max Power (Analytic)       : {:1.5f}".format(max_anal))
    #print("Relative Max Diff (EX) [%] : {:1.5f}".format(RDM_diff_EXP[-1]))


p = Plotter(max_time*1E3/timesteps,  xlabel=r"time step [ms]", ylabel=r"$L_2(p_{analytic} -  p_{epk})/L_2(p_{analaytic})$ [%]")
p.addData(100*np.array(l2_diff_CN), marker="kx", label="Crank-Nicholson")
p.addData(100*np.array(l2_diff_IMP), marker="yx", label="Implicit", alpha=0.4)
#p.addData(np.array(l2_diff_EXP), marker="gx", label="Explicit", alpha=0.4)
p.addData(np.ones((len(RDM_diff_CN),)) , label="1% Error")
p.save("./results/time_step.pdf")

p = Plotter(max_time * 1E3/timesteps,  xlabel=r"time step [ms]", ylabel=r"$(p^{max}_{analytic} -  p^{max}_{epk})/p^{max}_{analaytic}$ [%]")
p.addData(100*np.array(RDM_diff_CN), marker="kx", label="Crank-Nicholson")
p.addData(100*np.array(RDM_diff_IMP), marker="gx",label="Implicit")
#p.addData(np.array(RDM_diff_EXP), marker="gx",label="Explicit", alpha=0.4)
p.addData(np.ones((len(RDM_diff_CN),)) , label="1% Error")
p.save("./results/time_step_RDM.pdf")
