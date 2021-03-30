#! /usr/bin/python3

import numpy as np

from solver import *
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate as si
'''
Problem D.2 on 551 HW 4
'''

test()

plt_logx=False

#load in PARCS data
parcs_data = pd.read_csv("PARCS.dat", sep = " ")


def interp(qty, tt):
    f = si.interp1d(parcs_data["Time"], parcs_data[qty])
    return f(tt)

#set up time mesh
start_time, end_time = parcs_data["Time"].iloc[0], parcs_data["Time"].iloc[-1]
dt = 1.0e-5
t = np.arange(start_time, end_time + dt, dt)

t = t[:t.size//200]
#t = parcs_data["Time"].values
#print(t.shape)

#creating a better tmest
#N = 20000
#t = np.zeros(N)
#t[:1000] = parcs_data["Time"].values[:1000]
#t[1000:] = np.linspace(t.max()+dt, end_time, N - 1000)


#set up precursors
lambda_precursor = np.array([0.0128, 0.0318, 0.119, 0.3181, 1.4027, 3.9286])
beff = np.array([0.02584, 0.152, 0.13908, 0.30704, 0.1102, 0.02584])/100
#lambda_precursor = np.average(lambda_precursor, weights = beff)
#beff = beff.sum()

#create Data instance
d = ConstantKineticsData()
d.beff = beff
d.gamma_D = -1.2*np.sum(beff)
d.lambda_H = 1.0
d.mgt = 2.6E-15
d.lambda_precursor = lambda_precursor
data = Data.buildFromConstant(d, t)
data.mgt = interp("Generation-Time", t)
data.f_fp = interp("Normalization-Factor", t)
data.gamma_D = data.gamma_D/(data.f_fp )

#create Rho instance
rho = ReactivityGrid(t, interp("Reactivity", t))
rho_PARCS = np.copy(rho.rho)
idx = np.argmax(rho.rho)
max_rho =  1.078 * np.sum(beff)
rho.rho *= max_rho/ rho.rho[idx]
rho.rho[idx:] = max_rho
rx_plt = Plotter(t, ylabel=r"$\rho(t)$ [\$]")
rx_plt.addData(rho.rho / np.sum(beff), logx=plt_logx, label=r"$\rho_{IM}$ [\$]")

#create the solver instance
solver = Solver(data, t, rho, debug=True)
p0 = interp("Relative-Power",t)[0]
solver.setIniCond(p0)
#import pdb
#pdb.run('solver.solve(0.5, feedback=True)')
solver.solve(0.5, feedback=True)
power_plt = Plotter(t)
zeta_plt = Plotter(t, ylabel=r"$\frac{\zeta(t)}{\zeta(0)}$ [a.u.]")

power_plt.addData(interp("Relative-Power",t) , label="PARCS", logx=plt_logx)
power_plt.addData(solver.p*interp("Normalization-Factor", t), label="EPK", logx=plt_logx)
power_plt.save("./results/d2a_power.pdf")

rx_plt.addData(solver.rho / np.sum(beff), logx=plt_logx, label=r"$EPK, \rho$ [\$]")
rx_plt.addData(rho_PARCS, logx=plt_logx, label=r"$PARCS, \rho$ [\$]")
rx_plt.save("./results/d2a_rx.pdf")

for i in range(len(beff)):
    zeta_plt.addData(np.copy(solver.zetas[i,:]/solver.zetas[i,0]), label="Group {}".format(i+1))
zeta_plt.save("./results/c1b_zeta.pdf")

plt.legend()
plt.show()
