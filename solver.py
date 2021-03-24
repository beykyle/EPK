#! /usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
import matplotlib


"""
This module numerically solves the 6-delayed group point kinetics equations with linear feedback
"""

def find(a : np.array , val : float):
    '''
    finds the index of the largest value in a that is less than val, assuming a is sorted, or 0
    if val is smaller than any value in a
    '''
    if   ( val > max(a) ): return len(a) - 1
    elif ( val < min(a) ): return 0
    return min(np.argwhere((a - val) > 0 )) - 1

def test():
    aa = np.array([0.01 , 1, 89,  100 ])
    assert(find(aa,0) == 0)
    assert(find(aa,0.99) == 0)
    assert(find(aa,1.001) == 1)
    assert(find(aa,90) == 2)

test()

class ConstantKineticsData:
    def __init__(self, precursor_groups : int, mgt=2.6E-5, f_fp=1.0, beff=0.76, gamma_D=0, lambda_H=0.0, ):
        self.mgt      = mgt
        self.f_fp     = f_fp
        self.beff     = beff
        self.gamma_D  = gamma_D
        self.lambda_H = lambda_H
        self.lambda_precursor = np.ones((precursor_groups,)) * 0.49405

class Data:
    def __init__(self, lambda_H ,  beff, mgt, gamma_D, f_fp, lambda_precursor, timesteps):
        self.lambda_H = lambda_H
        self.beff = beff
        self.mgt = mgt
        self.gamma_D = gamma_D
        self.f_fp = f_fp
        self.lambda_precursor = lambda_precursor
        self.precursor_groups = lambda_precursor.shape[0]
        self.timesteps = timesteps
        assert(lambda_H.shape == (timesteps,))
        assert(beff.shape     == (timesteps,))
        assert(mgt.shape      == (timesteps,))
        assert(gamma_D.shape  == (timesteps,))
        assert(f_fp.shape     == (timesteps,))
        assert(lambda_precursor.shape == (self.precursor_groups,))

    def collapseGroups(self):
        if precursor_groups == 1:
            return self

        new_lambda = np.ones((1,self.timesteps))
        for i in range(0,self.timesteps):
            new_lambda[i] = np.mean(self.lambda_precursor[:,i])

        return Data(self.lambda_H, self.beff, self.mgt,
                    self.gamma_D , self.f_fp, self.new_lambda, timesteps)

    @classmethod
    def buildFromConstant(self, d : ConstantKineticsData, time_grid : np.array):
        grid_shape = time_grid.shape
        precursor_grid_shape = d.lambda_precursor.shape
        return Data( np.ones(grid_shape) * d.lambda_H ,
                     np.ones(grid_shape) * d.beff ,
                     np.ones(grid_shape) * d.mgt ,
                     np.ones(grid_shape) * d.gamma_D ,
                     np.ones(grid_shape) * d.f_fp ,
                     np.ones(precursor_grid_shape) * d.lambda_precursor ,
                     time_grid.shape[0])


class Reactivty:
    def rho(self):
        return 0

    def analyticPower1DG(self, data_1dg : Data, time, p0 : float):
        print("Analytic solution not available for general reactivity insertion")
        exit(1)

class LinearReactivityRamp(Reactivty):
    def __init__(self, rho_0 : float, rho_f : float, time : float):
        self.rho_s = rho_0
        self.rho_dot = (rho_f - rho_0)/time

    def rho(self, t : float):
        return rho_s + t * rho_dot

    def analyticPower1DG(self, data_1dg : Data, p0 : float, time : np.array, i0, iff):
        dt = time[i0:iff] - time[i0]
        l   = data_1dg.lambda_precursor[0]
        tau = (data_1dg.beff[i0:iff]  - self.rho_s)/ self.rho_dot
        C   = data_1dg.beff[i0:iff] * l / self.rho_dot + 1
        return p0 * np.exp(- l * dt) * ( tau / ( tau - dt))**(C)

class PieceWiseReactivityRamp(Reactivty):
    def __init__(self, times : list, reactivities : list):
        self.reactivities = reactivities
        self.times = np.array(times)

    def rho(self, t : float):
        if (t < self.times[0] or t > self.times[-1]): return 0
        return reactivities[find(self.times, t)].rho(t)

    def analyticPower1DG(self, data_1dg : Data, t : np.array, p0 : float):
        # make sure reactivity ramp happens on a larger time scale than the time step
        max_dt = np.max(t[1:-1] - t[0:-2])
        max_ramp_step = np.max(self.times[1:-1] - self.times[0:-2])
        assert(max_dt < max_ramp_step)

        # set up power
        p = np.zeros(t.shape)

        # solve in each piecewise section
        for i,r in enumerate(self.reactivities):
            i0  =  find(t,self.times[i])[0]
            iff =  find(t,self.times[i+1])[0] + 2
            p[i0:iff] = r.analyticPower1DG(data_1dg, p0, t, i0, iff)
            p0 = p[iff-1]

        return p


class Solver:
    def __init__(self, data : Data, time : np.array, reactivity : Reactivty):
        self.d = data
        self.t = time
        self.timesteps = time.size
        assert(self.t.shape == (self.timesteps,))
        self.reactivity = reactivity
        # initialize arrays for output quantities
        self.H = np.zeros(time.shape)
        self.p = np.zeros(time.shape)
        self.zetas = np.zeros((self.timesteps,data.precursor_groups))
        # set initial conditions
        self.p[0] = 1
        self.H[0] = self.p[0] * self.d.f_fp[0]
        self.zetas[0,:] = 1/(self.d.lambda_precursor) * self.d.beff[0] * self.p[0]

    def solve(self, theta):
        #TODO
        pass

    def analyticPower1DG(self):
        if ( np.any(self.d.lambda_H != 0) or np.any(self.d.gamma_D != 0) ):
            print("Analytic solutions only implemented without feedback")
            exit(1)

        data_1dg = self.d
        if ( (self.d.precursor_groups != 1) ):
            data_1dg = self.d.collapseGroups()

        return self.reactivity.analyticPower1DG(data_1dg, self.t, self.p[0])

class Plotter:
    def __init__(self, time : np.array, xlabel=r"$t$ [s]", ylabel=r"$\frac{p(t)}{p(0)}$ [a.u.]"):
        self.t = time
        self.fig = plt.figure(figsize=(12, 6))
        self.ax = plt.axes()

        self.font = { 'family': 'serif',
                      'color':  'black',
                      'weight': 'regular',
                      'size': 12,
                      }

        self.title_font = { 'family': 'serif',
                            'color':  'black',
                            'weight': 'bold',
                            'size': 12,
                          }

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)


    def addData(self, data : np.array, label : str):
        self.ax.plot(self.t, data, label=label)

    def save(self, fname: str):
        self.ax.legend()
        self.fig.savefig(fname)

    def plotReactivityRamp(self, rho : PieceWiseReactivityRamp):
        pass
