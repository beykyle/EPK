#! /usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
import matplotlib


"""
This module numerically solves the 6-delayed group point kinetics equations with linear feedback
"""
def findNearestIDX(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()

class ConstantKineticsData:
    def __init__(self, precursor_groups : int, mgt=2.6E-5, f_fp=1.0 beff=0.76, gamma_D=0, lambda_H=0.0, ):
        self.mgt      = mgt
        self.f_fp     = f_fp
        self.beff     = beff
        self.gamma_D  = gamma_D
        self.lambda_H = lambda_H
        self.lambda_precursor = np.ones((1,precursor_groups)) * 0.49405

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
        assert(lambda_H.shape = (1,timesteps))
        assert(beff.shape     = (1,timesteps))
        assert(mgt.shape      = (1,timesteps))
        assert(gamma_D.shape  = (1,timesteps))
        assert(f_fp.shape     = (1,timesteps))
        assert(lambda_precursor.shape[1] = (timesteps))

    @classmethod
    def buildFromConstant(d : ConstantKineticsData, time_grid : np.array):
        grid_shape = time_grid.shape()
        precursor_grid_shape = d.lambda_precursor.shape
        return Data( np.ones(grid_shape) * d.lambda_H ,
                     np.ones(grid_shape) * d.beff ,
                     np.ones(grid_shape) * d.mgt ,
                     np.ones(grid_shape) * d.gamma_D ,
                     np.ones(grid_shape) * d.f_fp ,
                     np.ones(precursor_grid_shape) * d.lambda_precursor ,
                     d.timesteps)

    def collapseGroups(self):
        if precursor_groups == 1:
            return self

        new_lambda = np.ones((1,self.timesteps))
        for i in range(0,self.timesteps):
            new_lambda[i] = np.mean(self.lambda_precursor[:,i])

        return Data(self.lambda_H, self.beff, self.mgt,
                    self.gamma_D , self.f_fp, self.new_lambda, timesteps)

class Reactivty:
    def rho(self):
        return 0

    def analyticPower1DG(self, data_1dg : Data, time):
        print("Analytic solution not available for general reactivity insertion")
        exit(1)

class LinearReactivityRamp(Reactivty):
    def __init__(self, rho_0 : float, rho_f : float, time : float):
        self.rho_s = rho_0
        self.rho_dot = (rho_f - rho_0)/time

    def rho(self, t : float):
        return rho_s + t * rho_dot

    def analyticPower1DG(self, data_1dg : Data, time : np.array, p0 : float):
        l   = data_1dg.lambda_precursor[0]
        tau = (data_1dg.beff  - self.rho_s)/ self.rho_dot
        C   = data_1dg.beff * l / self.rho_dot
        return p0 * np.exp(- l * time) * ( tau / ( tau - time))**(-C)

class PieceWiseReactivityRamp(Reactivty):
    def __init__(self, times : list, reactivities : list):
        self.reactivities = reactivities
        self.times = np.array(times)

    def rho(self, t : float):
        if (t < times[0] or t > times[-1]) return 0
        return reactivities[findNearestIDX(self.times, t)].rho(t)

    def analyticPower1DG(self, data_1dg : Data, t : np.array, p0 : float):
        p = np.zeros(t.shape)
        p[0] = p0
        for i,r in enumerate(self.reactivities):
            ramp_start =  time[i]
            ramp_stop  =  time[i+1]
            i0  =  findNearestIDX(t,ramp_start)
            iff =  findNearestIDX(t,ramp_stop)
            p[i0:iff] = r.analyticPower1DG(data_1dg, time[i0:iff], p[i0])
        return p


class Solver:
    def __init__(self, data : Data, time : np.array, reactivity : Reactivty):
        self.d = data
        self.t = time
        self.timesteps = self.time.size
        assert(self.t.shape = (1,self.timesteps))
        self.reactivity = reactivity
        # initialize arrays for output quantities
        self.H = np.zeros(time.shape)
        self.p = np.zeros(time.shape)
        self.zetas = np.zeros((6,self.timesteps))

    def solve(self, theta):
        #TODO
        pass

    def analyticPower1DG(self):
        if ( (self.data.lambda_H != 0) or (self.data.gamma_D != 0) ):
            print("Analytic solutions only implemented without feedback")
            exit(1)
        if ( (seld.data.precursor_groups != 1) ):
            print("Analytic solutions only implemented for a single precursor group")
        power = np.zeros((1,self.timesteps))
        #TODO
        return power



class Plotter:
    def __init__(time, xlabel=r"$t$ [s]", ylabel=r"$\frac{p(t)}{p(0)}$ [a.u.]"):
        self.t = time
        self.xlabel = xlabel
        font = {'family' : 'normal',
                'size'   : 18}
        matplotlib.rc('font', **font)
        #TODO set up fig

    def addPower(data : np.array):
        pass

    def save(self, fname):
        pass

    @classmethod
    def plotReactivityRamp(rho : PieceWiseReactivityRamp):
        pass
