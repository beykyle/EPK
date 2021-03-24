#! /usr/bin/python3

from solver import ConstantKineticsData, Data, Solver, LinearReactivityRamp, PieceWiseReactivityRamp

# set up time grid
timesteps = 10
t = np.linspace(0,10,num=timesteps)

# set up kinetics data
precursor_groups = 1
d = ConstantKineticsData(precursor_groups)
d.beff = 0.76
d.mgt = 2.6E-15
d.lambda_precursor  = d.lambda_precursor * 0.49405

# thermal feedback dynamics data

# build data object gridded over time steps
data = Data.buildFromConstant(d)

# build a reactivity ramp
times = [0,1,5]
rho_ramp_up = LinearReactivityRamp(0,0.5 * beff, 1)
rho_ramp_down = LinearReactivityRamp(0.5 * beff, 0, 5)
rho = PieceWiseReactivityRamp(times , [rho_ramp_up, rho_ramp_down])

# run the solver
solver.solve(0.5)
power_numeric = solver.p

# get the analytic solution
solver = Solver(data,t,rho)
power_analytic = solver.analyticSolve1DG()

# plot reactivity ramp
Plotter.plotReactivityRamp(rho)

# compare analytic and numeric power with the plotter
plotter = Plotter(t)
plotter.addPower(power_analytic, "Analytic")
plotter.addPower(power_numeric,  "EPKE Solver")
plotter.save("./p1.pdf")
