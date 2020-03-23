###############################################################################
###############################################################################
### General Imports
###############################################################################
###############################################################################
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# General packages
import warnings; warnings.simplefilter(action='ignore', category=FutureWarning)
import os

# The meat and potatoes: this allows us to call R in python
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri; pandas2ri.activate()

# Graphing packages
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Data manipulation packages
import pandas as pd
import numpy as np
import statsmodels.api as sm

###############################################################################
###############################################################################
### Plotting of RD Design
###############################################################################
###############################################################################
def rdplot(y, x, df, covs = None, residualize = False, x_range = [], c = 0, p = 4, nbins = None, binselect = 'esmv',
           scale = None, kernel = 'uni', weights = None, h = None, support = None, subset = None,
           hide = False, R_options = None, verbose = False, size = True, legend = False):
    '''
Implements several data-driven Regression Discontinuity (RD) plots, using either evenly-spaced or quantile-spaced partitioning. Two type of RD plots are constructed: (i) RD plots with binned sample means tracing out the underlying regression function, and (ii) RD plots with binned sample means mimicking the underlying variability of the data. See here: https://www.rdocumentation.org/packages/rdrobust/versions/0.99.4/topics/rdplot.

Inputs:
    y is the dependent variable. It should be a string representing a column in your dataframe.

    x is the running variable (a.k.a. score or forcing variable). It should also be a string column.

    df specifies the pandas dataframe where this data is coming from

    covs specifies additional covariates to be used in the polynomial regression. They should be a list of columns names in your dataframe (strings).

    residualize uses the residual of the LHS with respect to the covariates instead of the LHS itself. Covariates need to be specified for this option to be True.

    x_range allows you to trim the dataframe. This is seperate from the R option "support" as it is crude and will just remove all observations outside the range of the running variable.

    c specifies the RD cutoff in x; default is c = 0.

    p specifies the order of the global-polynomial used to approximate the population conditional mean functions for control and treated units; default is p = 4.

    nbins specifies the number of bins used to the left of the cutoff, denoted J−, and to the right of the cutoff, denoted J+, respectively. If not specified, J+ and J− are estimated using the method and options chosen below.

    binselect specifies the procedure to select the number of bins. This option is available only if J− and J+ are not set manually. Options are

        es: IMSE-optimal evenly-spaced method using spacings estimators.

        espr: IMSE-optimal evenly-spaced method using polynomial regression.

        esmv: mimicking variance evenly-spaced method using spacings estimators. This is the default option.

        esmvpr: mimicking variance evenly-spaced method using polynomial regression.

        qs: IMSE-optimal quantile-spaced method using spacings estimators.

        qspr: IMSE-optimal quantile-spaced method using polynomial regression.

        qsmv: mimicking variance quantile-spaced method using spacings estimators.

        qsmvpr: mimicking variance quantile-spaced method using polynomial regression.

    scale  specifies a multiplicative factor to be used with the optimal numbers of bins selected. Specifically, the number of bins used for the treatment and control groups will be scale×^J+ and scale×^J−, where ^J⋅ denotes the estimated optimal numbers of bins originally computed for each group; default is scale = 1.

    kernel  specifies the kernel function used to construct the local-polynomial estimator(s). Options are: triangular, epanechnikov, and uniform. Default is kernel=uniform (i.e., equal/no weighting to all observations on the support of the kernel).

    weights  is the variable used for optional weighting of the estimation procedure. The unit-specific weights multiply the kernel function.

    h  specifies the bandwidth used to construct the (global) polynomial fits given the kernel choice kernel. If not specified, the bandwidths are chosen to span the full support of the data. If two bandwidths are specified, the first bandwidth is used for the data below the cutoff and the second bandwidth is used for the data above the cutoff.

    support  specifies an optional extended support of the running variable to be used in the construction of the bins; default is the sample range.

    subset is optional vector specifying a subset of observations to be used

    hide supresses the graph (running it often causes the program to fail)

    R_options should be a string. It allows you to manually insert additional options (see R documentation of the original program for more information). You should insert any options as a string, in the same format as you would in R. However, this should not really be used. Most of these options are for formatting R plots, which are default going to be turned off here.

    verbose has it print the rdplot call from R for you

    size (if true) means that your scatterplot will have dots that grow larger with the mass at that point.

    legend only matters if size is turned on. This should be either 'brief', 'full', or False

Output:
    ax which is the plot axis

    binselect method used to compute the optimal number of bins.

    N sample sizes used to the left and right of the cutoff.

    Nh effective sample sizes used to the left and right of the cutoff.

    c cutoff value.

    p order of the global polynomial used.

    h bandwidth used to the left and right of the cutoff.

    kernel kernel used.

    J selected number of bins to the left and right of the cutoff.

    J_IMSE IMSE optimal number of bins to the left and right of the cutoff.

    J_MV Mimicking variance number of bins to the left and right of the cutoff.

    coef matrix containing the coefficients of the pth order global polynomial estimated both sides of the cutoff.

    vars_bins data frame containing the variables used to construct the bins: bin id, cutoff values, mean of x and y within each bin, cutoff points and confidence interval bounds.

    vars_poly data frame containing the variables used to construct the global polynomial plot.

    scale selected scale value.

    rscale implicit scale value.

    bin_avg average bin length.

    bin_med median bin length.

    text_rdplot_arg the actual R code used to call the underlying function.
    '''
    # Pull in the dictionary of all the available variables
    d = vars()
    df = df.copy()

    if covs and type(covs)==str:
        covs = [covs]

    # General all purpose roptions
    roptions = ['c','p','nbins','binselect','scale','kernel','weights','h','support']

    # y and x are called every time
    rdplot_call = 'y=df$%s,' %y + 'x=df$%s,' %x + 'hide=TRUE'

    # Covariates only called if they exist, and then we treat them separately for list or string
    if residualize==True:
        if covs:
            Z = df.filter(covs)
        else:
            raise ValueError('Specified residuals but no valid covariates')
        # Now getting the covariates, adding constant and executing residualization
        Z = sm.add_constant(Z)
        # residualize the LHS and bring in the residuals into the frame
        df =  df.join(sm.OLS(df[y],Z,missing='drop').fit().resid.to_frame(name='res_%s' %y))
        # y will now be the residuals we just loaded
        df[y] = df['res_%s' %y]
    # if not residualize, just deal with covariates normally if they exist
    elif covs and type(covs)==list:
        all_covars = ',covs=cbind(' + ','.join(['df$%s' %x for x in covs])
        rdplot_call += all_covars+')'
    # Take a subset
    if subset:
        rdplot_call += ','+'subset='+'df$%s' %subset

    for key in roptions:
        value = d[key]
        if type(value) == str:
            rdplot_call += ''.join([',',key,'=',"'%s'" %value])
        elif type(value) == list:
            rdplot_call += ''.join([',',key,'=c(',''.join([str(x)+',' for x in value])[:-1],')'])
        elif type(value) == bool:
            rdplot_call += ''.join([',',key,'=',str(value).upper()])
        elif type(value) == type(None):
            rdplot_call += ''.join([',',key,'=NULL'])
        else:
            rdplot_call += ''.join([',',str(key),'=',str(value)])
    if R_options:
        rdplot_call += ',' + R_options

    function_call = '\n'.join(filter(None,
                                     ['library(rdrobust)',
                                      "df = read.csv('temp_file_for_rdplot.csv')",
                                      "out = rdplot(%s)" %rdplot_call]))

    if x_range:
        df = df.loc[df[x].between(x_range[0],x_range[1]),
                    [x,y] + covs if covs else [x,y]]
    else:
        df = df.loc[:,[x,y] + covs if covs else [x,y]]
    # Modifications to the dataframe to get rid of infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.to_csv('temp_file_for_rdplot.csv')
    out = ro.r(function_call)
    if verbose:
        print(out)
    os.remove('temp_file_for_rdplot.csv')

    elements = dict()
    for t in out.names:
        if t=='vars_poly' or t=='vars_bins':
            temp_elements = dict()
            for s in out.rx2(t).names:
                temp_elements[s] = ro.pandas2ri.ri2py(out.rx2(t).rx2(s))
            elements[t] = pd.DataFrame(temp_elements)
        else:
            try:
                elements[t] = ro.pandas2ri.ri2py(out.rx2(t))
            except:
                print("There was an error, probably with importing R code")
                pass

    line_output = elements['vars_poly']
    bin_output = elements['vars_bins']
    bin_output = bin_output.rename(columns={'rdplot_N':'Obs'})

    class rd_dict:
        # So we get nice output._____ entries, this is bad coding
        def __init__(self, **entries):
            self.__dict__.update(entries)

    if hide:
        result = rd_dict(text_rdplot_arg=rdplot_call,**elements)
    else:
        print('RD of %s on %s' %(y,x))
        if size:
            ax = sns.scatterplot(x='rdplot_mean_bin',y='rdplot_mean_y',
                                 data=bin_output,s=75,size='Obs',legend=legend)
        else:
            ax = sns.scatterplot(x='rdplot_mean_bin',y='rdplot_mean_y',
                                 data=bin_output,s=75)

        plt.axvline(c,color=sns.color_palette()[1],linewidth=.75)
        sns.lineplot(x=line_output.rdplot_x[line_output.rdplot_x<0],y=line_output.rdplot_y[line_output.rdplot_x<0],
                     ax=ax,color=sns.color_palette()[0],linewidth=2)
        sns.lineplot(x=line_output.rdplot_x[line_output.rdplot_x>0],y=line_output.rdplot_y[line_output.rdplot_x>0],
                     ax=ax,color=sns.color_palette()[0],linewidth=2)
        ax.set(xlabel=x,ylabel=y)
        # Result includes the plot axis
        result = rd_dict(ax=ax,text_rdplot_arg=rdplot_call,**elements)

    return result


###############################################################################
###############################################################################
### Robust RD Estimation
###############################################################################
###############################################################################
def rdrobust(y, x, df, covs=None, x_range=[], c = 0, fuzzy = None, deriv = 0, p = 1, q=2, h = None,
             bwselect = 'mserd', vce = 'nn', cluster = None, nnmatch = 3, level = 95, b = None,
             rho = None, kernel = 'tri', weights = None, scalepar = 1, scaleregul = 1,
             sharpbw = False, rep_all = True, subset = None, verbose=True):
    '''
Implements local polynomial Regression Discontinuity (RD) point estimators with robust bias-corrected confidence intervals and inference procedures. See here: https://www.rdocumentation.org/packages/rdrobust/versions/0.99.4/topics/rdrobust.

Inputs:
    y is the dependent variable.

    x is the running variable (a.k.a. score or forcing variable).

    c specifies the RD cutoff in x; default is c = 0.

    x_range allows you to trim the dataframe. This is seperate from the R option "support" as it is crude and will just remove all observations outside the range of the running variable.

    verbose (if True) displays the underlying R summary o the RD results.

    fuzzy specifies the treatment status variable used to implement fuzzy RD estimation (or Fuzzy Kink RD if deriv=1 is also specified). Default is Sharp RD design and hence this option is not used.

    deriv specifies the order of the derivative of the regression functions to be estimated. Default is deriv=0 (for Sharp RD, or for Fuzzy RD if fuzzy is also specified). Setting deriv=1 results in estimation of a Kink RD design (up to scale), or Fuzzy Kink RD if fuzzy is also specified.

    p specifies the order of the local-polynomial used to construct the point-estimator; default is p = 1 (local linear regression).

    q specifies the order of the local-polynomial used to construct the bias-correction; default is q = 2 (local quadratic regression).

    h specifies the main bandwidth used to construct the RD point estimator. If not specified, bandwidth h is computed by the companion command rdbwselect. If two bandwidths are specified, the first bandwidth is used for the data below the cutoff and the second bandwidth is used for the data above the cutoff.

    b specifies the bias bandwidth used to construct the bias-correction estimator. If not specified, bandwidth b is computed by the companion command rdbwselect. If two bandwidths are specified, the first bandwidth is used for the data below the cutoff and the second bandwidth is used for the data above the cutoff.

    rho specifies the value of rho, so that the bias bandwidth b equals h/rho. Default is rho = 1 if h is specified but b is not.

    covs specifies additional covariates to be used for estimation and inference.

    kernel is the kernel function used to construct the local-polynomial estimator(s). Options are triangular (default option), epanechnikov and uniform.

    weights is the variable used for optional weighting of the estimation procedure. The unit-specific weights multiply the kernel function.

    bwselect specifies the bandwidth selection procedure to be used. By default it computes both h and b, unless rho is specified, in which case it only computes h and sets b=h/rho.

    vce specifies the procedure used to compute the variance-covariance matrix estimator (Default is vce=nn). Options are

        nn for heteroskedasticity-robust nearest neighbor variance estimator with nnmatch the (minimum) number of neighbors to be used.

        hc0 for heteroskedasticity-robust plug-in residuals variance estimator without weights.

        hc1 for heteroskedasticity-robust plug-in residuals variance estimator with hc1 weights.

        hc2 for heteroskedasticity-robust plug-in residuals variance estimator with hc2 weights.

        hc3 for heteroskedasticity-robust plug-in residuals variance estimator with hc3 weights.

    cluster indicates the cluster ID variable used for cluster-robust variance estimation with degrees-of-freedom weights. By default it is combined with vce=nn for cluster-robust nearest neighbor variance estimation. Another option is plug-in residuals combined with vce=hc0.

    nnmatch to be combined with for vce=nn for heteroskedasticity-robust nearest neighbor variance estimator with nnmatch indicating the minimum number of neighbors to be used. Default is nnmatch=3

    level sets the confidence level for confidence intervals; default is level = 95.

    scalepar specifies scaling factor for RD parameter of interest. This option is useful when the population parameter of interest involves a known multiplicative factor (e.g., sharp kink RD). Default is scalepar = 1 (no scaling).

    scaleregul specifies scaling factor for the regularization term added to the denominator of the bandwidth selectors. Setting scaleregul = 0 removes the regularization term from the bandwidth selectors; default is scaleregul = 1.

    sharpbw option to perform fuzzy RD estimation using a bandwidth selection procedure for the sharp RD model. This option is automatically selected if there is perfect compliance at either side of the cutoff.

    rep_all if specified, rdrobust reports three different procedures

        conventional RD estimates with conventional standard errors.

        bias-corrected estimates with conventional standard errors.

        bias-corrected estimates with robust standard errors.

    subset an optional vector specifying a subset of observations to be used.

Output:
    N vector with the sample sizes used to the left and to the right of the cutoff.

    N_h vector with the effective sample sizes used to the left and to the right of the cutoff.
    c cutoff value.

    p order of the polynomial used for estimation of the regression function.

    q order of the polynomial used for estimation of the bias of the regression function.

    bws matrix containing the bandwidths used.

    tau_cl conventional local-polynomial estimate to the left and to the right of the cutoff.

    tau_bc bias-corrected local-polynomial estimate to the left and to the right of the cutoff.

    coef vector containing conventional and bias-corrected local-polynomial RD estimates.

    se vector containing conventional and robust standard errors of the local-polynomial RD estimates.

    bias estimated bias for the local-polynomial RD estimator below and above the cutoff.

    beta_p_l conventional p-order local-polynomial estimates to the left of the cutoff.

    beta_p_r conventional p-order local-polynomial estimates to the right of the cutoff.

    V_cl_l conventional variance-covariance matrix estimated below the cutoff.

    V_cl_r conventional variance-covariance matrix estimated above the cutoff.

    V_rb_l robust variance-covariance matrix estimated below the cutoff.

    V_rb_r robust variance-covariance matrix estimated above the cutoff.

    pv vector containing the p-values associated with conventional, bias-corrected and robust local-polynomial RD estimates.

    ci matrix containing the confidence intervals associated with conventional, bias-corrected and robust local-polynomial RD estimates.
    '''
    # Pull in the dictionary of all the available variables
    d = vars()
    df = df.copy()

    if covs and type(covs)==str:
        covs = [covs]

    # General all purpose roptions
    roptions = ['c','deriv','p','q','h','bwselect','vce',
                'nnmatch','level','b','rho','kernel','weights','scalepar',
                'scaleregul','sharpbw']

    # y and x are called every time
    rdplot_call = 'y=df$%s,' %y + 'x=df$%s' %x

    # Other variables are actually in the dataframe and need to be added to the call
    if covs and type(covs)==list:
        all_covars = ',covs=cbind(' + ','.join(['df$%s' %x for x in covs])
        rdplot_call += all_covars+')'

    # Take a subset
    if subset:
        rdplot_call += ','+'subset='+'df$%s' %subset
    if fuzzy:
        rdplot_call += ','+'fuzzy='+'df$%s' %fuzzy
    if cluster:
        rdplot_call += ','+'cluster='+'df$%s' %cluster
    if rep_all:
        # all was changed to rep_all so as not to conflict with pythonic syntax
        rdplot_call += ','+'all=TRUE'

    # The rest are options to be added as needed depending on the type, they are translated to R
    for key in roptions:
        value = d[key]
        if type(value) == str:
            rdplot_call += ''.join([',',key,'=',"'%s'" %value])
        elif type(value) == list:
            rdplot_call += ''.join([',',key,'=c(',''.join([str(x)+',' for x in value])[:-1],')'])
        elif type(value) == bool:
            rdplot_call += ''.join([',',key,'=',str(value).upper()])
        elif type(value) == type(None):
            rdplot_call += ''.join([',',key,'=NULL'])
        else:
            rdplot_call += ''.join([',',str(key),'=',str(value)])

    function_call = '\n'.join(filter(None,
                                     ['library(rdrobust)',
                                      "df = read.csv('temp_file_for_rdplot.csv')",
                                      "out = rdrobust(%s)" %rdplot_call]))
    if x_range:
        df = df[df[x].between(x_range[0],x_range[1])]
    # Modifications to the dataframe to get rid of infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.to_csv('temp_file_for_rdplot.csv')
    out = ro.r(function_call)
    os.remove('temp_file_for_rdplot.csv')

    elements = dict()
    for t in [x for x in out.names if x!='all']:
        try:
            elements[t] = ro.pandas2ri.ri2py(out.rx2(t))
        except:
            print("There was an error, probably with importing R code")
            print("Failed on %s" %t)
            pass

    class rd_dict:
        # So we get nice output._____ entries, this is bad coding
        def __init__(self, **entries):
            self.__dict__.update(entries)

    result = rd_dict(text_rdplot_arg=rdplot_call,**elements)
    printout = pd.DataFrame(np.concatenate((result.coef,result.se,result.z,result.pv),axis=1),
                            columns=['Coef.','Std. Error','z','p>z']).join(
                                    pd.DataFrame(result.ci,columns=['95% Lower','95% Upper']))
    printout.index = ['Conventional','Bias-Corrected','Robust']
    result = rd_dict(text_rdplot_arg=rdplot_call,printout=printout,**elements)

    if verbose:
        print(out)
        print('RD of %s on %s' %(y,x))
        print(np.round(printout,3))

    return result

###############################################################################
###############################################################################
### Bandwidth Selection for RD Design
###############################################################################
###############################################################################
def rdbwselect(y, x, df, covs=[], x_range=[], c = 0, fuzzy = None, deriv = 0, p = 1, q=2,
             bwselect = 'mserd', vce = 'nn', cluster = None, nnmatch = 3,
             kernel = 'tri', weights = None, scaleregul = 1,
             sharpbw = False, rep_all = False, subset = None, verbose=True):
    '''
Implements bandwidth selectors for local polynomial Regression Discontinuity (RD) point estimators and inference procedures. See here: https://www.rdocumentation.org/packages/rdrobust/versions/0.99.4/topics/rdbwselect.

Inputs:
    y is the dependent variable.

    x is the running variable (a.k.a. score or forcing variable).

    c specifies the RD cutoff in x; default is c = 0.

    x_range allows you to trim the dataframe. This is seperate from the R option "support" as it is crude and will just remove all observations outside the range of the running variable.

    verbose (if True) displays the underlying R summary o the RD results.

    fuzzy specifies the treatment status variable used to implement fuzzy RD estimation (or Fuzzy Kink RD if deriv=1 is also specified). Default is Sharp RD design and hence this option is not used.

    deriv specifies the order of the derivative of the regression functions to be estimated. Default is deriv=0 (for Sharp RD, or for Fuzzy RD if fuzzy is also specified). Setting deriv=1 results in estimation of a Kink RD design (up to scale), or Fuzzy Kink RD if fuzzy is also specified.

    p specifies the order of the local-polynomial used to construct the point-estimator; default is p = 1 (local linear regression).

    q specifies the order of the local-polynomial used to construct the bias-correction; default is q = 2 (local quadratic regression).

    covs specifies additional covariates to be used for estimation and inference.

    kernel is the kernel function used to construct the local-polynomial estimator(s). Options are triangular (default option), epanechnikov and uniform.

    weights is the variable used for optional weighting of the estimation procedure. The unit-specific weights multiply the kernel function.

    bwselect specifies the bandwidth selection procedure to be used. Options are

        mserd one common MSE-optimal bandwidth selector for the RD treatment effect estimator.

        msetwo two different MSE-optimal bandwidth selectors (below and above the cutoff) for the RD treatment effect estimator.

        msesum one common MSE-optimal bandwidth selector for the sum of regression estimates (as opposed to difference thereof).

        msecomb1 for min(mserd,msesum).

        msecomb2 for median(msetwo,mserd,msesum), for each side of the cutoff separately.

        cerrd one common CER-optimal bandwidth selector for the RD treatment effect estimator.

        certwo two different CER-optimal bandwidth selectors (below and above the cutoff) for the RD treatment effect estimator.

        cersum one common CER-optimal bandwidth selector for the sum of regression estimates (as opposed to difference thereof).

        cercomb1 for min(cerrd,cersum).

        cercomb2 for median(certwo,cerrd,cersum), for each side of the cutoff separately.

        Note: MSE = Mean Square Error; CER = Coverage Error Rate. Default is bwselect=mserd. For details on implementation see Calonico, Cattaneo and Titiunik (2014a), Calonico, Cattaneo and Farrell (2018), and Calonico, Cattaneo, Farrell and Titiunik (2017), and the companion software articles.

    vce specifies the procedure used to compute the variance-covariance matrix estimator (Default is vce=nn). Options are

        nn for heteroskedasticity-robust nearest neighbor variance estimator with nnmatch the (minimum) number of neighbors to be used.

        hc0 for heteroskedasticity-robust plug-in residuals variance estimator without weights.

        hc1 for heteroskedasticity-robust plug-in residuals variance estimator with hc1 weights.

        hc2 for heteroskedasticity-robust plug-in residuals variance estimator with hc2 weights.

        hc3 for heteroskedasticity-robust plug-in residuals variance estimator with hc3 weights.

    cluster indicates the cluster ID variable used for cluster-robust variance estimation with degrees-of-freedom weights. By default it is combined with vce=nn for cluster-robust nearest neighbor variance estimation. Another option is plug-in residuals combined with vce=hc0.

    nnmatch to be combined with for vce=nn for heteroskedasticity-robust nearest neighbor variance estimator with nnmatch indicating the minimum number of neighbors to be used. Default is nnmatch=3

    scaleregul specifies scaling factor for the regularization term added to the denominator of the bandwidth selectors. Setting scaleregul = 0 removes the regularization term from the bandwidth selectors; default is scaleregul = 1.

    sharpbw option to perform fuzzy RD estimation using a bandwidth selection procedure for the sharp RD model. This option is automatically selected if there is perfect compliance at either side of the threshold.

    rep_all if specified, rdbwselect reports all available bandwidth selection procedures.

    subset an optional vector specifying a subset of observations to be used.

Output:
    N vector with sample sizes to the left and to the righst of the cutoff.

    c cutoff value.

    p order of the local-polynomial used to construct the point-estimator.

    q order of the local-polynomial used to construct the bias-correction estimator.

    bws matrix containing the estimated bandwidths for each selected procedure.

    bwselect bandwidth selection procedure employed.

    kernel kernel function used to construct the local-polynomial estimator(s).
   '''
    # Pull in the dictionary of all the available variables
    d = vars()
    df = df.copy()
    # General all purpose roptions
    roptions = ['c','deriv','p','q','bwselect','vce',
                'nnmatch','kernel','weights',
                'scaleregul','sharpbw']

    # y and x are called every time
    rdplot_call = 'y=df$%s,' %y + 'x=df$%s' %x

    # Other variables are actually in the dataframe and need to be added to the call
    if covs and type(covs)==list:
        all_covars = ',covs=cbind(' + ','.join(['df$%s' %x for x in covs])
        rdplot_call += all_covars[:-1]+')'
    elif covs and type(covs)==str:
        all_covars = ',covs=df$%s' %covs
    # Take a subset
    if subset:
        rdplot_call += ','+'subset='+'df$%s' %subset
    if fuzzy:
        rdplot_call += ','+'fuzzy='+'df$%s' %fuzzy
    if cluster:
        rdplot_call += ','+'cluster='+'df$%s' %cluster
    if rep_all:
        # all was changed to rep_all so as not to conflict with pythonic syntax
        rdplot_call += ','+'all=TRUE'

    # The rest are options to be added as needed depending on the type, they are translated to R
    for key in roptions:
        value = d[key]
        if type(value) == str:
            rdplot_call += ''.join([',',key,'=',"'%s'" %value])
        elif type(value) == list:
            rdplot_call += ''.join([',',key,'=c(',''.join([str(x)+',' for x in value])[:-1],')'])
        elif type(value) == bool:
            rdplot_call += ''.join([',',key,'=',str(value).upper()])
        elif type(value) == type(None):
            rdplot_call += ''.join([',',key,'=NULL'])
        else:
            rdplot_call += ''.join([',',str(key),'=',str(value)])

    function_call = '\n'.join(filter(None,
                                     ['library(rdrobust)',
                                      "df = read.csv('temp_file_for_rdplot.csv')",
                                      "out = rdbwselect(%s)" %rdplot_call]))
    if x_range:
        df = df[df[x].between(x_range[0],x_range[1])]
    # Modifications to the dataframe to get rid of infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.to_csv('temp_file_for_rdplot.csv')
    out = ro.r(function_call)
    os.remove('temp_file_for_rdplot.csv')

    elements = dict()
    for t in [x for x in out.names if x!='all']:
        try:
            elements[t] = ro.pandas2ri.ri2py(out.rx2(t))
        except:
            print("There was an error, probably with importing R code")
            print("Failed on %s" %t)
            pass

    class rd_dict:
        # So we get nice output._____ entries, this is bad coding
        def __init__(self, **entries):
            self.__dict__.update(entries)

    result = rd_dict(text_rdplot_arg=rdplot_call,**elements)
    printout = pd.DataFrame(data=result.bws,index=result.bw_list,
                            columns=['BW est. (h) Left','BW est. (h) R','BW bias. (b) Left','BW bias. (b) Right'])
    result = rd_dict(text_rdplot_arg=rdplot_call,printout=printout,**elements)

    if verbose:
        print(out)
        print('RD of %s on %s' %(y,x))
        print(np.round(printout,2))

    return result
