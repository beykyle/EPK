#! /usr/bin/python3

import numpy as np

from solver import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
'''
Problem C.1 on 551 HW 4
'''

#
test()

# set up time grid
max_time = 6.00
timesteps = 151
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
tmax = 1.0 # max time of reactivity ramp
times = [0,tmax,max_time] # total asym ramp time: 6s
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
step_size = np.array([1.0E-3, 10E-3, 20E-3, 25E-3, 40E-3, 50E-3])
timesteps = np.round(max_time/step_size).astype(dtype=int) +1
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
    solver.solve(0.5, feedback=False)
    max_CN = np.max(solver.p)
    p.addData(np.copy(solver.p), label="CN", marker="+")
    l2_diff_CN.append(L2(power_analytic,solver.p))
    RDM_diff_CN.append( np.fabs((max_anal - max_CN)/max_anal) )
    solver.reset()
    # implicit
    solver.solve(1.0, feedback=False)
    max_IM = np.max(solver.p)
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
    print("Relative Max Diff (CN) [%] : {:1.5f}".format(100*RDM_diff_CN[-1]) )
    print("Relative Max Diff (IM) [%] : {:1.5f}".format(100*RDM_diff_IMP[-1]))
    print("Max Power (Analytic)       : {:1.5f}".format(max_anal))
    #print("Relative Max Diff (EX) [%] : {:1.5f}".format(RDM_diff_EXP[-1]))

pct_error = 100*np.array(RDM_diff_CN)
dt_ms = max_time * 1E3/timesteps

p = Plotter(dt_ms,  xlabel=r"$\Delta t$ [ms]", ylabel=r"$L_2(p_{analytic} -  p_{epk})/L_2(p_{analaytic})$ [%]")
p.addData(100*np.array(l2_diff_CN), marker="+", label="Crank-Nicholson")
p.addData(100*np.array(l2_diff_IMP), marker="x", label="Implicit", alpha=0.4)
#p.addData(np.array(l2_diff_EXP), marker="gx", label="Explicit", alpha=0.4)
p.addData(np.ones((len(RDM_diff_CN),))*0.01 , label="0.01% Error")
p.save("./results/time_step.pdf")


p = Plotter(dt_ms,  xlabel=r"$\Delta t$ [ms]", ylabel=r"$(p^{max}_{analytic} -  p^{max}_{epk})/p^{max}_{analaytic}$ [%]")
p.addData(100*np.array(RDM_diff_CN), marker="+", label="Crank-Nicholson")
p.addData(100*np.array(RDM_diff_IMP), marker="x",label="Implicit")
#p.addData(np.array(RDM_diff_EXP), marker="gx",label="Explicit", alpha=0.4)
p.addData(np.ones((len(RDM_diff_CN),))*0.01 , label="0.01% Error")


# curve fitting
def quadratic(dt,a,b):
    return a*dt**2 + b*dt

popt, pcov = curve_fit(quadratic,dt_ms,pct_error)
lbl = "{:1.3f}".format(popt[0]) + "[ms$^{-2}$]$(\Delta t)^2$ + "\
        + "{:1.4f}".format(popt[1]) + " [ms$^{-1}$]$\Delta t$"
p.addData(quadratic(dt_ms, *popt), label=lbl)
p.save("./results/time_step_RDM.pdf")

# find dt such that pct_err == 0.01%
c = -0.01
a = popt[0]
b = popt[1]
det = b**2 - 4 * a *c
assert(det>0)
assert(a>0)
optimal_dt =  (np.sqrt(det) - b)/(2*a)
print("="*50)
print("\n\nOptimal timestep [ms]: {}".format(optimal_dt))



## Part C.1.b
print("\nRunning asymmetric ramp problem with optimal timestep...")
optimal_dt_s = optimal_dt * 10E-3
timesteps = np.round(max_time/optimal_dt_s).astype(dtype=int) +1
t = np.linspace(0,max_time,num=timesteps)
power_plt = Plotter(t)
zeta_plt = Plotter(t, ylabel=r"$\frac{\zeta(t)}{\zeta(0)}$ [a.u.]")
rx_plt = Plotter(t, ylabel=r"$\rho(t)$ [\$]")
power_plt = Plotter(t)
d.mgt = 2.6E-5
data = Data.buildFromConstant(d,t)
rho = PieceWiseReactivityRamp(times , [rho_ramp_up, rho_ramp_down], t)
s = Solver(data,t,rho)
s.solve(0.5)
power_plt.addData(s.p, label=r"EPK, $\theta=\frac{1}{2}$", marker="+")
power_plt.addData(np.copy(s.analyticPower1DG()), label="Analytic", marker="-")
power_plt.save("./results/c1b_power.pdf")
rx_plt.plotReactivityRamp(rho,data.beff)
rx_plt.save("./results/c1b_rx.pdf")
zeta_plt.addData(np.copy(s.zetas[0,:]/s.zetas[0,0]), label="Group 1")
zeta_plt.save("./results/c1b_zeta.pdf")


## part C.1.c
print("\nEvaluating analytic PJA for <1s 0.5$ insertion...")
# get times less than 1s
d2 = d
d2.mgt = 2.6E-15
s2 = Solver(Data.buildFromConstant(d2,t),t,rho)
s2.solve(0.5)
idx = find(t,1.0)
t = t[0:idx+1]
power_plt  = Plotter(t)
power_plt.addData(s.p[0:idx+1],  label=r"EPK, $\Lambda = $2.6E-5s", marker="+")
power_plt.addData(s2.p[0:idx+1],  label=r"EPK, $\Lambda = $2.6E-15s", marker="+")
power_plt.addData(s2.analyticPower1DG()[0:idx+1], label="PJA", marker="-")
power_plt.save("./results/c1d_power.pdf")

pa = s2.analyticPower1DG()[0:idx+1]
pdiff_plt  = Plotter(t, ylabel=r"$\frac{p(t) - p_{analytic}(t)}{p_{analytic}(t)}$ [%]")
pdiff_plt.addData(100*(pa - s.p[0:idx+1] )/pa,  label=r"EPK, $\Lambda = $2.6E-5s", marker="+")
pdiff_plt.addData(100*(pa - s2.p[0:idx+1])/pa,  label=r"EPK, $\Lambda = $2.6E-15s", marker="+")
pdiff_plt.save("./results/c1d_diff.pdf")
