#! /usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.pyplot as mpl
mpl.rcParams['font.size'] = 16


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


def get1Gbeff(beff : np.array):
    return np.sum(beff)

def betaWeightedLambda(beff : np.array , lambda_precursor : np.array):
    return np.dot(beff , lambda_precursor) / get1Gbeff(beff)

def invBetaWeightedLambda(beff : np.array, lambda_precursor : np.array):
    return get1Gbeff(beff) / np.dot(beff , 1.0/lambda_precursor)

def rho2Dollars(beff : np.array, rho : np.array):
    assert(beff.shape[1] == rho.size)
    beff1G = np.zeros(beff.shape[1])
    for i in range(beff.shape[1]):
        beff1G[i] = get1Gbeff(beff[:,i])
    return rho / beff1G

class ConstantKineticsData:
    def __init__(self, mgt=2.6E-5, f_fp=1.0, gamma_D=0, lambda_H=0.0,beff=np.array([0.0076]), lambda_precursor=np.array([0.49405])):
        self.mgt              = mgt
        self.f_fp             = f_fp
        self.gamma_D          = gamma_D
        self.lambda_H         = lambda_H
        self.beff             = beff
        self.lambda_precursor = lambda_precursor
        assert(self.lambda_precursor.size == self.beff.size)
        self.precursor_groups = self.lambda_precursor.size

class Data:
    def __init__(self, lambda_H, mgt, gamma_D, f_fp, lambda_precursor, beff, timesteps):
        self.lambda_H = lambda_H
        self.mgt = mgt
        self.gamma_D = gamma_D
        self.f_fp = f_fp
        self.lambda_precursor = lambda_precursor
        self.precursor_groups = lambda_precursor.shape[0]
        self.beff =  beff
        self.timesteps = timesteps
        assert(lambda_H.shape == (timesteps,))
        assert(mgt.shape      == (timesteps,))
        assert(gamma_D.shape  == (timesteps,))
        assert(f_fp.shape     == (timesteps,))
        assert(beff.shape             == (self.precursor_groups, timesteps))
        assert(lambda_precursor.shape == (self.precursor_groups, timesteps))

    def collapseGroups(self, beta_weighted=True):
        if self.precursor_groups == 1:
            return self

        new_lambda = np.zeros((1,self.timesteps))
        new_beff   = np.zeros((1,self.timesteps))
        for i in range(self.timesteps):
            new_beff[0,i] = get1Gbeff(self.beff[:,i])
            if beta_weighted:
                new_lambda[0,i] = betaWeightedLambda(self.beff[:,i],self.lambda_precursor[:,i])
            else:
                new_lambda[0,i] = invBetaWeightedLambda(self.beff[:,i],self.lambda_precursor[:,i])

        return Data(self.lambda_H, self.mgt, self.gamma_D , self.f_fp,
                new_lambda, new_beff, self.timesteps)

    def print_all(self):
        print("f_fp")
        print(self.f_fp)
        print("Lambda_H")
        print(self.lambda_H)
        print("Gamma_D")
        print(self.gamma_D)
        print("Lambda_precursor")
        print(self.lambda_precursor)
        print("beff")
        print(self.beff)
        print("mgt")
        print(self.mgt)

    @classmethod
    def buildFromConstant(self, d : ConstantKineticsData, time_grid : np.array):
        grid_shape = time_grid.shape
        precursor_grid_shape = (d.lambda_precursor.size,time_grid.size)
        beff = np.zeros(precursor_grid_shape)
        lambda_precursor = np.zeros(precursor_grid_shape)

        for i in range(time_grid.size):
            beff[:,i] = d.beff
            lambda_precursor[:,i] = d.lambda_precursor

        return Data( np.ones(grid_shape) * d.lambda_H ,
                     np.ones(grid_shape) * d.mgt ,
                     np.ones(grid_shape) * d.gamma_D ,
                     np.ones(grid_shape) * d.f_fp ,
                     lambda_precursor,
                     beff,
                     time_grid.shape[0])

class Reactivity:
    pass

class ReactivityGrid(Reactivity):
    def __init__(self, t : np.array, rho : np.array):
        assert(t.shape == rho.shape)
        self.t = t
        self.rho = rho

class LinearReactivityRamp():
    def __init__(self, rho_0_dollars : float, rho_f_dollars : float, time : float):
        self.rho_s = rho_0_dollars
        self.rho_dot = (rho_f_dollars  - rho_0_dollars )/time

    def analyticPower1DG(self, data_1dg : Data, p0 : float, time : np.array, i0, iff):
        dt = time[i0:iff] - time[i0]
        l   = data_1dg.lambda_precursor[0,i0:iff]
        if (self.rho_dot != 0):
            beff = data_1dg.beff[0,i0:iff]
            tau = (beff  - self.rho_s)/ self.rho_dot
            C   = beff * l / self.rho_dot + 1
            return p0 * np.exp(- l * dt) * ( tau / ( tau - dt))**(C)
        else:
            return p0 * np.exp(l * self.rho_s * dt/(beff - self.rho_s))

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
            if i != len(self.reactivities) - 1:
                p0 = p[iff-1]

        return p

    def getRhoGrid(self):
        rho_grid = np.zeros(self.t.shape)
        for i,r in enumerate(self.reactivities):
            iff =  find(self.t,self.times[i+1]) + 1
            i0  =  find(self.t,self.times[i])
            rho_grid[i0:iff] = r.getRhoGrid(self.t[i0:iff])

        return rho_grid

def k0(x : float):
    return 1. - x/2 + x**2/6  - x**3/24 + x**4/120- x**5/720 + x**6/5040

def k1(x : float):
    return 0.5 - x/6 + x**2/24  - x**3/120 + x**4/720- x**5/5040 + x**6/40320

class Solver:
    def __init__(self, data : Data, time : np.array, reactivity : Reactivity,
                 debug=False, time_dep_precurs = False):
        self.debug = debug
        self.time_dep_precurs = time_dep_precurs
        self.d = data
        self.t = time
        self.reactivity = reactivity

        self.k0 = lambda x: k0(x)
        self.k1 = lambda x: k1(x)

        self.reset()

    def setIniCond(self, p0:float):
        self.p[0] = p0
        self.H[0] = self.p[0] * self.d.f_fp[0]
        self.zetas[:,0] = self.d.beff[:,0] * self.p[0] / self.d.lambda_precursor[:,0]

    def resetNewData(self, d, t, p0=1):
        # initialize arrays for output quantities
        self.d       = d
        self.t       = t
        self.dt = self.t[1:] - self.t[:-1] # time step
        #assert(np.sum(self.dt) == self.t[-1] - self.t[0])
        self.timesteps = self.t.size
        assert(self.t.shape == (self.timesteps,))
        assert(self.t.shape == self.reactivity.t.shape)
        self.H       = np.zeros(self.t.shape)
        self.G       = np.zeros((self.d.precursor_groups, self.timesteps))
        self.S       = np.zeros(self.t.shape)
        self.Shat    = np.zeros(self.t.shape)
        self.p       = np.zeros(self.t.shape)
        self.zetas   = np.zeros((self.d.precursor_groups,self.timesteps))
        self.rho_im  = np.copy(self.reactivity.rho)
        self.rho     = self.rho_im
        # set initial conditions
        self.setIniCond(p0)
        if (self.debug):
            print("Data")
            self.d.print_all()
            print("Reactivity")
            print(self.rho)
            print("Imposed reactivity")
            print(self.rho_im)

    def reset(self):
        self.resetNewData(self.d,self.t)

    def stepPowerFeedback(self, theta, alpha, tau_n, n):
        # get a1,b1
        lambda_H_hat = self.d.lambda_H[n] * self.dt[n-1]
        lambda_H_tilde = (self.d.lambda_H[n] + alpha)*self.dt[n-1]
        a1 = self.d.f_fp[n]*self.d.gamma_D[n]*self.dt[n-1]*self.k1(lambda_H_tilde)

        rho_d_n = self.rho[n] - self.rho_im[n]
        rho_d_nm1 = self.rho[n-1] - self.rho_im[n-1]
        P0 = self.p[0]#
        b1 = self.rho_im[n] + np.exp(-lambda_H_hat)*rho_d_nm1 - P0*self.d.gamma_D[n]*self.dt[n-1] \
                *self.k0(lambda_H_hat) + np.exp(alpha*self.dt[n-1])*self.d.gamma_D[n]*self.dt[n-1]\
                *self.d.f_fp[n-1]*self.p[n-1]*(self.k0(lambda_H_tilde)-self.k1(lambda_H_tilde))

        # get a,b,c
        beta_n = get1Gbeff(self.d.beff[:,n])
        beta_nm1 = get1Gbeff(self.d.beff[:,n-1])
        a = theta*self.dt[n-1]*a1 / self.d.mgt[n]
        temp = (b1 - beta_n)/self.d.mgt[n] - alpha
        b = theta*self.dt[n-1]*(temp  + tau_n / self.d.mgt[0]) - 1
        temp = (self.rho[n-1] - beta_nm1)/self.d.mgt[n-1] - alpha
        c = theta*self.dt[n-1]/self.d.mgt[0]*self.Shat[n] + np.exp(alpha*self.dt[n-1])\
                *((1-theta)*self.dt[n-1]*(temp*self.p[n-1] + self.S[n-1]/self.d.mgt[0]) \
                + self.p[n-1])

        # solve quadratic for new power
        det = b**2 - 4*a*c
        assert(a<=0)
        assert(det>0)
        if np.isclose(a,0):
            p = -c/b
        elif a < 0:
            p = -(b + np.sqrt(det))/(2*a)
        rho = a1 * p + b1
        #p = p/self.d.f_fp[n]
        #print(p.shape)
        #exit()
        return p,rho

    def stepPower(self, theta : float, alpha : float, tau_n : float, n : int):
        beff_nm1 = get1Gbeff(self.d.beff[:,n-1])
        beff_n   = get1Gbeff(self.d.beff[:,n])
        num = np.exp(alpha * self.dt[n-1]) \
            * (self.p[n-1] + (1-theta) * self.dt[n-1]  \
                * (((self.rho_im[n-1] - beff_nm1)/self.d.mgt[n-1] - alpha) \
                       * self.p[n-1]  + self.S[n-1] / self.d.mgt[0] \
                  ) \
              ) \
            + theta * self.dt[n-1] * self.Shat[n] / self.d.mgt[0]
        den = 1 - theta * self.dt[n-1] \
            * ((self.rho_im[n] - beff_n)/self.d.mgt[n] - alpha + tau_n/self.d.mgt[0])

        p = num/den
        rho = self.rho_im[n]
        return p, rho

    def solve(self, theta, feedback=True):
        print("Running Solver w/ theta={}".format(theta))
        if feedback:
            step = self.stepPowerFeedback
        else:
            step = self.stepPower

        if (self.debug):
            print("n\tt [s]\tdt   \ta_n  \tl_t  \tz_n  \trho_n\tp_n  ")

        for n in range(1,self.t.size):

            if n % 5000 == 0 and self.t.size > 100000:
                print(n, self.t.size)

            if (n > 1) and (not np.isclose(self.p[n-2],0.) ):
                alpha = 1/self.dt[n-2]*np.log(self.p[n-1]/self.p[n-2])
                gamma = self.dt[n-2]/self.dt[n-1]
            else:
                alpha = 0
                gamma = 1

            #calculate omega and zeta
            lambda_tilde = (self.d.lambda_precursor[:,n] + alpha) * self.dt[n-1]
            omega = self.d.mgt[0]/self.d.mgt[n] * self.d.beff[:,n] * self.dt[n-1] \
                  * k1(lambda_tilde)
            self.G[:,n-1] = self.d.mgt[0]/self.d.mgt[n-1]*self.d.beff[:,n]*self.p[n-1]
            zeta_hat = np.exp(-self.d.lambda_precursor[:,n]*self.dt[n-1])*self.zetas[:,n-1] \
                    + np.exp(alpha*self.dt[n-1]) * self.dt[n-1] * self.G[:,n-1] \
                     * (k0(lambda_tilde) - k1(lambda_tilde))
            if self.time_dep_precurs:
                self.d.lambda_precursor[:, n] = np.dot(zeta_hat, self.d.lambda_precursor[:, 0])\
                        /zeta_hat.sum()


            # calculate tau_n, Shat_n, S_(n-1)
            tau_n = np.dot(self.d.lambda_precursor[:,n] , omega)
            self.Shat[n] =  np.dot(self.d.lambda_precursor[:,n] , zeta_hat)
            self.S[n-1] =  np.dot(self.d.lambda_precursor[:,n-1] , self.zetas[:,n-1])

            # calculate new power
            pnew, rhonew = step(theta, alpha, tau_n, n)

            # test if exp transform gives better convergence than linear
            if (n > 1):
                if ( np.fabs(pnew - np.exp(alpha * self.dt[n-1]) * self.p[n-1] ) <=
                     np.fabs(pnew - self.p[n-1] - (self.p[n-1] - self.p[n-2])/gamma ) ):
                    self.p[n] = pnew
                    self.rho[n] = rhonew
                else:
                    alpha = 0
                    self.p[n], self.rho[n] = step(theta, 0, tau_n, n)
            else:
                self.p[n] = pnew


            # evaluate new H, G, rho and zetas
            self.H[n] = self.d.f_fp[n] * self.p[n]
            self.zetas[:,n] = self.p[n] * omega + zeta_hat

            # print debug time step info
            if(self.debug):
                print("{}\t{:6.4f}\t{:6.4f}\t{:6.4f}\t{:6.4f}\t{:6.4f}\t{:6.4f}\t{:6.4f}"
                        .format(n, self.t[n], self.dt[n-1], alpha, lambda_tilde[0],
                                self.zetas[0,n], self.rho[n], self.p[n])
                )

    def analyticPower1DG(self, beta_weighted=True):
        # analytic soln only w/out feedback
        assert(not np.any(self.d.lambda_H != 0))
        assert(not np.any(self.d.gamma_D != 0))

        # collapse to 1G
        data_1dg = self.d.collapseGroups(beta_weighted=beta_weighted)

        return self.reactivity.analyticPower1DG(data_1dg, self.p[0])

class Plotter:
    def __init__(self, time : np.array, xlabel=r"$t$ [s]", ylabel=r"$\frac{p(t)}{p(0)}$ [a.u.]"):
        self.t = time
        self.fig = plt.figure(figsize=(10, 6))
        self.ax = plt.axes()

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)


    def addData(self, data : np.array, label=None, marker="-", alpha=1., log=False, logx=False):
        d  = np.copy(data)
        if label != None:
            if log:
                self.ax.loglog(self.t, d, marker, label=label, alpha=alpha, linewidth=2.2, markersize=12)
            if logx:
                self.ax.semilogx(self.t, d, marker, label=label, alpha=alpha, linewidth=2.2, markersize=12)
            else:
                self.ax.plot(self.t, d, marker, label=label, alpha=alpha, linewidth=2.2, markersize=12)
        else:
            if log:
                self.ax.loglog(self.t, d, marker, alpha=alpha, linewidth=2.2, markersize=12)
            if logx:
                self.ax.semilogx(self.t, d, marker, alpha=alpha, linewidth=2.2, markersize=12)
            else:
                self.ax.plot(self.t, d, marker, alpha=alpha, linewidth=2.2, markersize=12)

    def save(self, fname: str):
        print("Saving figure: " + fname)
        self.ax.legend()
        plt.tight_layout()
        self.fig.savefig(fname)

    def plotReactivityRamp(self, rho : PieceWiseReactivityRamp, beff : np.array, label=None, marker="-", alpha=1.):
        self.addData( rho2Dollars( beff, rho.rho) , label=label, marker=marker, alpha=alpha)

def test():
    aa = np.array([0.01 , 1, 89,  100 ])
    assert(find(aa,0) == 0)
    assert(find(aa,0.99) == 0)
    assert(find(aa,1.001) == 1)
    assert(find(aa,90) == 2)
    assert(find(aa,89.000001) == 2)
    assert(find(aa,89) == 2)
    assert(find(aa,100) == 3)
