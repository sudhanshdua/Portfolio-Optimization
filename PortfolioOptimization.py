# Portfolio Optimization Library
# 
# Contents:
#    Part 1 (Risk and Return metrics)
#       ->  Annualized volatility 
#       ->  Annualized returns
#       ->  Compounded rate of return
#       ->  Skewness
#       ->  Kurtosis
#       ->  Sharpe ratio
#       ->  Semi-deviation, 
#       ->  Is Normal (Checking the normality of the series)
#       ->  VaR
#       ->  CVaR
#       ->  Parametric VaR
#       ->  Portfolio return
#       ->  Portfolio volatility
#       ->  Drawdown  (Computing the drawdown for the return data)
#       ->
#       ->
#       ->
#       ->
#       ->


###################               Modules               ###################


import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm
from scipy.optimize import minimize
import math


###################              Risk and Return              #####################


def annualize_vol(rets, N):
    """
    Annualizes the volatility (vol) for a set of returns
    
    Arguments:
        rets - return data
        N    - No. of periods per year (for example: N = 12 for a monthly return data)
    """
    return rets.std() * (N ** 0.5)




def annualize_rets(rets, N):
    """
    Annualizes a set of returns (rets)
    
    Arguments:
        rets - return data (DataFrame)
        N    - No. of periods per year (for example: N = 12 for a monthly return data)
    """
    compounded_growth = (1+rets).prod()
    # compounded growth = (1 + ret_1), (1 + ret_2), ....... , (1 + ret_n)
    n = rets.shape[0]
    
    # annualized return = (compounded growth)**(1/n)   ----->   if the returns entered are annual returns,     i.e. No. of periods = 1
    # annualized return = (compounded growth)**(N/n)   ----->   if the returns entered have          --->           No. of periods per year = N
    return compounded_growth**(N/n)-1




def compound(rets):
    """
    Compounded return value from a set of returns
    
    Arguments:
        rets - return data
    """
    
    # [((1 + ret_1) * (1 + ret_2) * ....... * (1 + ret_n)) ^ (1 / n)] - 1
    #
    #                                 or
    #
    # [e ^ (log(1 + ret_1) + log(1 + ret_2) + ..... + log(1 + ret_n))] - 1
    return np.expm1(np.log1p(rets).sum())




def skewness(rets):
    """
    Returns the skewness of the supplied Series or DataFrame
    
    Alternative method: scipy.stats.skew()
    
    Arguments:
        rets - return data
    """
    return ((rets - rets.mean())**3).mean() / rets.std(ddof = 0)**3 




def kurtosis(rets):
    """
    Returns the skewness of the supplied Series or DataFrame
    
    Alternative method: scipy.stats.kurtosos()
    
    Arguments:
        rets - return data
    """
    return ((rets - rets.mean())**4).mean() / rets.std(ddof = 0)**4 




def sharpe_ratio(rets, rfr, N):
    """
    Computes the annualized sharpe ratio for a set of returns
    
    Arguments:
        rets - return data
        rfr  - annualized risk-free rate
        N    - No. of periods per year (for example: N = 12 for a monthly return data)
    """
    # risk-free rate(per period) = [(1 + risk-free rate(annual))^(1 / no. of periods)] - 1
    rf_per_period= (1+ rfr)**(1/periods_per_year)-1
    
    excess_return = rets - rf_per_period
    
    # calculating the annualized excess return using the method defined above
    ann_excess_return = annualize_rets(excess_return, N)
    
    # calculating the annualized volatility using the method defined above
    ann_vol = annualize_vol(excess_return, N)
    
    return ann_excess_return/ann_vol




def semideviation(rets):
    """
    Returns the negative semideviation of returns
    
    Argument:
    rets - return data (must be a Series or a DataFrame)
    """
    # return data is a series
    if isinstance(rets, pd.Series):
        # filtering the index values with the negative returns
        neg_rets = rets < 0
        return rets[neg_rets].std(ddof = 0)
    
    # return data is a dataframe
    elif isinstance(rets, pd.DataFrame):
        return rets.aggregate(semideviation)
    
    else:
        raise TypeError("Expected rets to be a Series or DataFrame")




def is_normal(rets, level = 0.01):
    """
    Determines if the Series is normal by applying the Jarque-Bera test
    Returns True or False
    
    default significance level = 1%     => confidence level = 99%
    
    Arguments:
        rets  - return data
        level - confidence level   --->  (default = 5%)
    """
    if isinstance(rets, pd.DataFrame):
        # create a series of all the return data present in the columns and apply the method on each series
        return rets.aggregate(is_normal)
    else:
        statistic, p_value = scipy.stats.jarque_bera(rets)
        return p_value > level



def var_historic(rets, level = 5):
    """
    Returns the historic VaR at a specified significance level
    
    Arguments:
        rets  - return data
        level - significance level   --->  (default = 5%)
    """
    if isinstance(rets, pd.DataFrame):
        # create a series of all the return data present in the columns and apply the method on each series
        return rets.aggregate(var_historic, level = level)  
    
    if isinstance(rets, pd.Series):
        return -np.percentile(rets, level)
    else:
        raise TypeError("Expected rets to be Series or DataFrame")
        
       
    
    
def cvar_historic(rets, level = 5):
    """
    Computes the Conditional VaR of a Series or DataFrame
    
    Arguments:
        rets  - return data
        level - significance level   --->  (default = 5%)
    """
    if isinstance(rets, pd.Series):
        # filtering the indices of the return values, in the series, that are less than or equal to Historic VaR
        is_beyond = rets <= -var_historic(rets, level = level)
        # calculating the average loss rate, calculate using the returns that are less than or equal to Historic VaR
        return -rets[is_beyond].mean()
    
    if isinstance(rets, pd.DataFrame):
        return rets.aggregate(cvar_historic, level = level)
    else:
        raise TypeError("Expected rets to be Series or DataFrame")
        



def var_gaussian(rets, level = 5, modified = False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    
    If "modified" is True, then the modified VaR is returned, using the Cornish-Fisher modification
    
    Arguments:
        rets     - return data
        level    - significance level   ---> (default = 5%)
        modified - True or False     ---> (default = False)
    """
    # compute the Z score assuming it a Gaussian distribution
    z = norm.ppf(level/100)
    
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        
        z = (z + (z**2 -1)*s/6 + (z**3 -3*z)*(k-3)/24 - (2*z**3 - 5*z)*(s**2/36))
       
    return -(rets.mean() + z * rets.std(ddof = 0))





def portfolio_return(weights, rets):
    """
    Computes the return of a portfolio based on the returns and the weights of the constituents
    
    Arguments:
        weights - weight vector (a numpy array or Nx1 matrix)
        rets    - return data   (a numpy array or Nx1 matrix)
    """
    
    # weights.T          -->    weight vector transpose
    # weights.T @ rets   -->    Dot product of the weight vector transpose and the return vector
    return weights.T @ rets




def portfolio_vol(weights, covariance_matrix):
    """
    Computes the volatility of a portfolio from a covariance matrix and constituents' weights
    
    Arguments:
        weights           - weight vector     (a numpy array or Nx1 matrix)
        covariance_matrix - covariance matrix (an N x N matrix)
    """
    
    # weights.T          -->    weight vector transpose
    #
    # 2-security portfolio variance = (w_1 * sigma_1)^2 + (w_2 * sigma)^2 + 2(w_1 * w_2 * cov(1,2))   ---->   where, sigma = variance ^ 0.5
    #
    # N-security portfolio variance = (weights.T @ covariance_matrix @ weights)
    return (weights.T @ covariance_matrix @ weights)**0.5




def drawdown(return_series: pd.Series):
    """
    Computes and returns a DataFrame that contains:
    -->  wealth  :  The level of wealth if $100 were invested at t = 0
    -->  previous peaks  :  The maximum level of wealth achieved during the time frame if $100 were invested at t = 0
    -->  percent drawdowns : The level of decline in the wealth throughout the time frame (drawdown = previous peaks - wealth)
    
    Argument:
    return_series - time series of asset returns
    """
    
    # wealth (t=0) = 100 * (1 + ret_1)
    # wealth (t=1) = 100 * (1 + ret_1) * (1 + ret_2)
    # ...
    # wealth (t=n) = 100 * (1 + ret_ 1) * (1 + ret_2) * ..... * (1 + ret_n)
    wealth = 100 * (1 + return_series).cumprod()
    
    
    # previous peaks finds the maximum values of the wealth during thet time frame (its graph can only go up and never go down)
    previous_peaks = wealth.cummax()
    
    percent_drawdown = (wealth / previous_peaks) - 1 
    
    return pd.DataFrame({
        "Wealth": wealth,
        "Peaks": previous_peaks,
        "Drawdown": percent_drawdown
    })

