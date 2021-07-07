import math, argparse
import numpy as np
import statsmodels.api as sm
from scipy.optimize import curve_fit
import pandas as pd
from scipy.stats import spearmanr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Function for fitting logistic functions to data')
    parser.add_argument('y', help='y-values of data points to fit')
    parser.add_argument('X', help='x-values corresponding to y values')
    parser.add_argument('to_perm_test', help='whether or not to run non-parametric permutation test, which assesses statistical significance of model fit')
    parser.add_argument('Nsim', help='# of null simulations to perform for permutation testing. 1000 or 10000 recommended')
    args = parser.parse_args()

    y = args.y
    X = args.X
    if args.to_perm_test is None:
        to_perm_test = False
    else:
        to_perm_test = args.to_perm_test
    if args.Nsim is None:
        Nsim = 1000

    main_log_fit(X, y, to_perm_test, Nsim)

# define log fn
def nl_func(x, c, k, l, b):
    return b + ( l/(1 + np.exp(-k * (x-c))) )

# main function
def main_log_fit(X, y, to_perm_test=False, Nsim=10000):

    modelfit = run_fit(y, X)

    if to_perm_test:
        print ('Perm test is selected')
        Z_true = modelfit.stats_z
        Z_null = np.zeros((Nsim))
        np.random.default_rng(2021)
        for isim in np.arange(0, Nsim):
            X_temp = np.random.permutation(X)
            modelfit_temp = run_fit(y, X_temp)
            Z_null[isim] = modelfit_temp.stats_z
            if np.remainder(isim, 100)==0:
                print('Simulation ' + str(isim))
        if Z_true > 0:
            P_perm = np.mean(Z_null >= Z_true) * 2
        elif Z_true < 0:
            P_perm = np.mean(Z_null <= Z_true) * 2
        else:
            P_perm = 1

        modelfit.stats_Pperm = P_perm

    return modelfit

# run model fit
def run_fit(y, X):
    # set up bounds for  model fit
    maxfev = 10000 # max number of iterations for curve_fit function
    y_min = np.min(y)
    y_max = np.max(y)

    # reasonable parameter limits, but may depend on the data 
    c_inflection_min = -10
    k_slope_min = 0
    k_slope_max = math.inf
    k_slope_initial = .1
    # x-change: -10:max(x)
    # slope (k): 0:2 [2 results in extremely steep curve given date range, reasonable max]
    # height (l): -2*min : 2*max
    # starting point: same as height

    baseline_x = np.mean(y[X < .2*np.max(X)])

    lower_bounds = [c_inflection_min, k_slope_min, y_min*2, y_min*2]
    upper_bounds = [np.max(X), k_slope_max, y_max*2, y_max*2]
    bounds_info = ( lower_bounds, upper_bounds )
    p0 = ([np.median(X), k_slope_initial, np.mean(y[X > .8*np.max(X)]), baseline_x])

    P_mfit = 1
    x = np.nan
    k = np.nan
    y_final = np.nan
    y_init = np.nan
    y_change = np.nan
    r_mfit = np.nan

    if np.isnan(y).sum() < len(y):
        try:
            popt, pcov = curve_fit(nl_func, X, y, maxfev=maxfev, bounds=bounds_info, p0=p0)
            y_pre = nl_func(X, popt[0], popt[1], popt[2], popt[3])

            x = popt[0]
            k = popt[1]
            y_init = popt[3]
            y_final = popt[2]
            y_change = y_final-y_init
            
            # one sided correlation test - negative fits have no meaning
            correl = spearmanr(y_pre, y)
            r_mfit = max(correl.correlation, 0)
            if r_mfit > 0:
                P_mfit = np.true_divide(correl.pvalue, 2)
            else:
                P_mfit = 1

            # encode correlation as negative if change is negative (decrease)
            r_mfit = r_mfit * np.sign(y_change)

        except RuntimeError:
            # nothing to do here 
            error=1
    else:
        # nothing to do here 
        error=1        

    class Container(object):
        pass

    modelfit = Container()
    modelfit.stats_r = r_mfit
    modelfit.stats_z = np.arctanh(r_mfit)
    modelfit.stats_Ppara = P_mfit
    modelfit.x = x
    modelfit.k = k
    modelfit.y_final = y_final
    modelfit.y_init = y_init
    modelfit.y_change = y_change
    if 'y_pre' in locals():
        modelfit.y_predict = y_pre
    return modelfit

