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
    if   ( val >= a.max() ): return a.size - 1
    elif ( val <= a.min() ): return 0
    return np.argwhere((a - val) > 0 ).min() - 1


class ConstantKineticsData:
    def __init__(self, mgt=2.6E-5, f_fp=1.0, beff=0.76, gamma_D=0, lambda_H=0.0, precursor_groups=1):
        self.mgt      = mgt
        self.f_fp     = f_fp
        self.beff     = beff
        self.gamma_D  = gamma_D
        self.lambda_H = lambda_H
        self.lambda_precursor = np.ones((precursor_groups,)) * 0.49405

    def precursor_groups():
        return self.lambda_precursor.shape[0]

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
        if self.precursor_groups == 1:
            return self

        #TODO weight with precursor group betas (or inverse)
        new_lambda = np.array([np.mean(self.lambda_precursor)])

        return Data(self.lambda_H, self.beff, self.mgt,
                    self.gamma_D , self.f_fp, new_lambda, self.timesteps)

    @classmethod
    def buildFromConstant(self, d : ConstantKineticsData, time_grid : np.array):
        grid_shape = time_grid.shape
        precursor_grid_shape = d.lambda_precursor.shape
        return Data( np.ones(grid_shape) * d.lambda_H ,
                     np.ones(grid_shape) * d.beff ,
                     np.ones(grid_shape) * d.mgt ,
                     np.ones(grid_shape) * d.gamma_D ,
                     np.ones(grid_shape) * d.f_fp ,
                     d.lambda_precursor ,
                     time_grid.shape[0])

class Reactivity:
    pass

class ReactivityGrid(Reactivity):
    def __init__(t : np.array, rho : np.array):
        assert(t.shape == rho.shape)
        self.t = t
        self.rho = rho

class LinearReactivityRamp():
    def __init__(self, rho_0_dollars : float, rho_f_dollars : float, time : float):
        self.rho_s = rho_0_dollars
        self.rho_dot = (rho_f_dollars  - rho_0_dollars )/time

    def analyticPower1DG(self, data_1dg : Data, p0 : float, time : np.array, i0, iff):
        dt = time[i0:iff] - time[i0]
        l   = data_1dg.lambda_precursor[0]
        tau = (data_1dg.beff[i0:iff]  - self.rho_s)/ self.rho_dot
        C   = data_1dg.beff[i0:iff] * l / self.rho_dot + 1
        return p0 * np.exp(- l * dt) * ( tau / ( tau - dt))**(C)

    def getRhoGrid(self, time : np.array ):
        return self.rho_s + self.rho_dot * (time - time[0])

class PieceWiseReactivityRamp(Reactivity):
    def __init__(self, times : list, reactivities : list, time_grid : np.array):
        self.reactivities = reactivities
        self.times = np.array(times)
        self.t = time_grid
        self.rho = self.getRhoGrid()

    def analyticPower1DG(self, data_1dg : Data, p0 : float):
        # make sure reactivity ramp happens on a larger time scale than the time step
        max_dt = np.max(self.t[1:-1] - self.t[0:-2])
        max_ramp_step = np.max(self.times[1:-1] - self.times[0:-2])
        assert(max_dt < max_ramp_step)

        # set up power
        p = np.zeros(self.t.shape)

        # solve in each piecewise section
        for i,r in enumerate(self.reactivities):
            i0  =  find(self.t,self.times[i])
            iff =  find(self.t,self.times[i+1]) + 2
            p[i0:iff] = r.analyticPower1DG(data_1dg, p0, self.t, i0, iff)
            p0 = p[iff-1]

        return p

    def getRhoGrid(self):
        rho_grid = np.zeros(self.t.shape)
        shift = 0
        for i,r in enumerate(self.reactivities):
            iff =  find(self.t,self.times[i+1]) + 1 + shift
            i0  =  find(self.t,self.times[i]) + shift
            shift = shift + 1
            rho_grid[i0:iff] = r.getRhoGrid(self.t[i0:iff])

        return rho_grid

class Solver:
    def __init__(self, data : Data, time : np.array, reactivity : Reactivity):
        self.d = data
        self.t = time
        self.dt = time[1:] - time[:-1] # time step
        self.timesteps = time.size
        assert(self.t.shape == (self.timesteps,))
        assert(self.t.shape == reactivity.t.shape)
        self.reactivity = reactivity
        self.rho_im = reactivity.rho
        # initialize arrays for output quantities
        self.H = np.zeros(time.shape)
        self.G = np.zeros(time.shape)
        self.p = np.zeros(time.shape)
        self.zetas = np.zeros((self.timesteps,data.precursor_groups))
        # set initial conditions
        self.p[0] = 1
        self.H[0] = self.p[0] * self.d.f_fp[0]
        self.G[0] = self.d.beff[0] * self.p[0]
        self.zetas[0,:] = 1/(self.d.lambda_precursor) * self.d.beff[0] * self.p[0]

        self.k0 = lambda x: (1 - np.exp(-x))/x
        self.k1 = lambda x: np.abs(1 - self.k0(x))/x

    #def step(self, theta, alpha, n):
            # perform quadratic precursor integration
            # calculate delayed source for time step
            # calculate H
            # handle feedback
        #zeta_hat = w*self.zetas[n-1] + w*self.d.mgt*self.p[n-1]*self.beff[n-1]/self.mgt \
        #        *(self.k0(self.lambda_precursor) - 
        #print(omega)
            #lambda_tilde = 
    def solve(self, theta):
        for n in range(1,self.t.size):
            # calculate alpha
            if n > 1:
                alpha = 1/self.dt[n-2]*np.log(self.p[n-1]/self.p[n-2])
                gamma = self.dt[n-2]/self.dt[n-1]
            else:
                alpha = 0
                gamma = self.dt[0]

            #calculate omega and zeta
            lambda_tilde = (self.d.lambda_precursor + alpha) * self.dt[n-1]
            omega = self.d.mgt[0]/self.d.mgt[n] * self.d.beff[n] * self.dt[n-1] * self.k1(lambda_tilde)
            zeta_hat = np.exp(-self.d.lambda_precursor[0]*self.dt[n-1])*self.zetas[n-1] \
                    + np.exp(alpha*self.dt[n-1]) * self.dt[n-1] * self.G[n-1] \
                    * (self.k0(lambda_tilde) - self.k1(lambda_tilde))

            pnew = self.step(theta, alpha, n)
            if ( (pnew - np.exp(alpha * self.dt[n-1]) * self.p[n-1] ) <=
                 (pnew - self.p[n-1] - (self.p[n-1] - self.p[n-2])/gamma ) ):
                self.p[n] = pnew
            else:
                self.p[n]  = self.step(theta,0,n)

            self.G[n] = self.d.mgt[0]/self.d.mgt[n] * self.d.beff[n] * self.p[n] \
                    * np.exp(-alpha*self.dt[n-1])

            # perform quadratic precursor integration
            # calculate delayed source for time step

    def analyticPower1DG(self):
        if ( np.any(self.d.lambda_H != 0) or np.any(self.d.gamma_D != 0) ):
            print("Analytic solutions only implemented without feedback")
            exit(1)

        data_1dg = self.d.collapseGroups()

        return self.reactivity.analyticPower1DG(data_1dg, self.p[0])

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


    def addData(self, data : np.array, label=None):
        if label != None:
            self.ax.plot(self.t, data, label=label)
        else:
            self.ax.plot(self.t, data)

    def save(self, fname: str):
        self.ax.legend()
        self.fig.savefig(fname)

    def plotReactivityRamp(self, rho : PieceWiseReactivityRamp):
        self.addData(rho.rho)

def test():
    aa = np.array([0.01 , 1, 89,  100 ])
    assert(find(aa,0) == 0)
    assert(find(aa,0.99) == 0)
    assert(find(aa,1.001) == 1)
    assert(find(aa,90) == 2)
    assert(find(aa,89.000001) == 2)
    assert(find(aa,89) == 2)
    assert(find(aa,100) == 3)

