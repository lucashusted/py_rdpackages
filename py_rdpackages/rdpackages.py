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
from io import StringIO
import sys

# The meat and potatoes: this allows us to call R in python
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri; pandas2ri.activate()

# Graphing packages
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set(font='Palatino')
sns.set_style('whitegrid',{'font':'Palatino','grid.linestyle': 'dotted'})

# Data manipulation packages
import pandas as pd
import numpy as np

###############################################################################
###############################################################################
### Plotting of RD Design
###############################################################################
###############################################################################
def rdplot(y, x, df, covs = None, dummies = None,
           c = 0, p = 4, nbins = None, binselect = 'esmv', covs_eval = 0,
           scale = None, kernel = 'uni', weights = None, h = None, ci = None,
           support = None, subset = None, hide = False, verbose = False,
           size = True, legend = False, covs_drop = True):
    '''
    Implements several data-driven Regression Discontinuity (RD) plots,
    using either evenly-spaced or quantile-spaced partitioning.
    Two type of RD plots are constructed: (i) RD plots with binned sample means
    tracing out the underlying regression function, and
    (ii) RD plots with binned sample means mimicking the underlying variability
    of the data. See here:
    https://www.rdocumentation.org/packages/rdrobust/versions/0.99.4/topics/rdplot.

    Parameters
    ----------
    y: the dependent variable. It should be a string representing
    a column in your dataframe.

    x: is the running variable (a.k.a. score or forcing variable).
    It should also be a string column.

    df: specifies the pandas dataframe where this data is coming from.

    covs: specifies additional covariates to be used in the polynomial regression.
    They should be a list of columns names in your dataframe (strings).

    dummies: specifies dummy variable covariates to be used in the regression.
    They should be a list of columns names in your dataframe (strings).
    Will automatically generate (dropping first) binary variables for each.

    c: specifies the RD cutoff in x; default is c = 0.

    p: specifies the order of the global-polynomial used to approximate
    the population conditional mean functions for control and treated units
    default is p = 4.

    nbins specifies the number of bins used to the left of the cutoff,
    denoted J−, and to the right of the cutoff, denoted J+, respectively.
    If not specified, J+ and J− are estimated using the method chosen below.

    binselect specifies the procedure to select the number of bins.
    This option is available only if J− and J+ are not set manually. Options are

        es: IMSE-optimal evenly-spaced method using spacings estimators.

        espr: IMSE-optimal evenly-spaced method using polynomial regression.

        esmv: mimicking variance evenly-spaced method using spacings estimators.
        This is the default option.

        esmvpr: mimicking variance evenly-spaced method using polynomial regression.

        qs: IMSE-optimal quantile-spaced method using spacings estimators.

        qspr: IMSE-optimal quantile-spaced method using polynomial regression.

        qsmv: mimicking variance quantile-spaced method using spacings estimators.

        qsmvpr: mimicking variance quantile-spaced method using polynomial regression.

    covs_eval: sets  the  evaluation  points  for  the  additional  covariates,
    when  included  in  the estimation. Options are: covs_eval = 0 (default)
    and covs_eval = 'mean'.

    scale:  specifies a multiplicative factor to be used with the optimal
    numbers of bins selected. Specifically, the number of bins used for the
    treatment and control groups will be scale×^J+ and scale×^J−, where ^J⋅
    denotes the estimated optimal numbers of bins originally computed
    for each group; default is scale = 1.

    kernel:  specifies the kernel function used to construct the
    local-polynomial estimator(s). Options are: triangular, epanechnikov,
    and uniform. Default is kernel=uniform (i.e., equal/no weighting to
    all observations on the support of the kernel).

    weights:  is the variable used for optional weighting of the estimation
    procedure. The unit-specific weights multiply the kernel function.

    h:  specifies the bandwidth used to construct the (global) polynomial fits
    given the kernel choice kernel. If not specified, the bandwidths are chosen
    to span the full support of the data. If two bandwidths are specified, the
    first bandwidth is used for the data below the cutoff and the second
    bandwidth is used for the data above the cutoff.

    support:  specifies an optional extended support of the running variable
    to be used in the construction of the bins; default is the sample range.

    subset: is optional vector specifying a subset of observations to be used,
    should be a string representing a column in the dataframe.

    hide: supresses the graph.

    verbose: has it print the rdplot call from R for you

    size: (if true) means that your scatterplot will have dots that grow larger
    with the mass at that point.

    legend: only matters if size is turned on.
    This should be either 'brief', 'full', or False

    covs_drop: In python, automatically drops any covariates that take on
    ONLY one value on the front end (fixing issue with original rdplot code)
    and also activates covs_drop in R which checks for other multicollinearity.

    Returns
    -------
    ax: which is the plot axis

    fig: which is the plot figure

    binselect: method used to compute the optimal number of bins.

    N: sample sizes used to the left and right of the cutoff.

    Nh: effective sample sizes used to the left and right of the cutoff.

    c: cutoff value.

    p: order of the global polynomial used.

    h: bandwidth used to the left and right of the cutoff.

    kernel: kernel used.

    J: selected number of bins to the left and right of the cutoff.

    J_IMSE: IMSE optimal number of bins to the left and right of the cutoff.

    J_MV: Mimicking variance number of bins to the left and right of the cutoff.

    coef: matrix containing the coefficients of the pth order global polynomial
    estimated both sides of the cutoff.

    vars_bins: data frame containing the variables used to construct the bins,
    bin id, cutoff values, mean of x and y within each bin,
    cutoff points and confidence interval bounds.

    vars_poly: data frame containing the variables used to construct
    the global polynomial plot.

    scale: selected scale value.

    rscale: implicit scale value.

    bin_avg: average bin length.

    bin_med: median bin length.

    text_rdplot_arg: the actual R code used to call the underlying function.
    '''
    # Pull in the dictionary of all the available variables
    d = vars()

    # Helper Classes (Bad Programming)
    class rd_dict:
        # So we get nice output._____ entries, this is bad coding
        def __init__(self, **entries):
            self.__dict__.update(entries)


    # At the very least, cut out missing values of the two main variables
    varlst = [x,y]
    df = df.dropna(subset=varlst).copy()
    dropped_vars = []

    # General all purpose options
    roptions = ['c','p','nbins','binselect','scale','kernel','weights','h',
                'support','covs_eval','covs_drop','ci']

    if covs:
        # convert singular covariates into a list for simplicity
        if type(covs)==str:
            covs = [covs]
        elif type(covs)==tuple:
            covs = list(covs)
        # dealing with smart removal
        if covs_drop:
            tempdf = df.loc[:,varlst].copy()
            for ii in covs:
                tempdf = tempdf.join(df[ii])
                if len(tempdf.dropna()[ii].unique())==1:
                    covs = [x for x in covs if x!=ii]
                    dropped_vars += [ii]
                    tempdf = tempdf.drop(columns=ii)
                tempdf = tempdf.dropna()
        # append the result regardless
        varlst += covs
    elif covs_eval != 0:
        raise ValueError('Specified cov_eval without covariates')

    df = df.dropna(subset=varlst).copy()

    if dummies:
        if type(dummies)==str:
            dummies = [dummies]

        for ii in dummies:
            tempdums = pd.get_dummies(df[ii],prefix='rddum_%s' %ii,drop_first=True)
            if tempdums.shape[1]>0:
                df = df.join(tempdums)
                if not covs:
                    covs = list(tempdums.columns)
                else:
                    covs = covs + list(tempdums.columns)
                varlst += list(tempdums.columns)


    # y and x are called every time
    rdplot_call = 'y=df$%s,' %y + 'x=df$%s,' %x + 'hide=TRUE'

    if covs:
        rdplot_call += ',covs=cbind(' + ','.join(['df$%s' %x for x in covs])+')'

    # Take a subset
    if subset:
        rdplot_call += ','+'subset='+'df$%s' %subset
        varlst += [subset]


    for key in roptions:
        value = d[key]
        if type(value) == str:
            rdplot_call += ''.join([',',key,'=',"'%s'" %value])
        elif type(value) == list:
            rdplot_call += ''.join([',',key,'=c(',''.join(
                [str(x)+',' for x in value])[:-1],')'])
        elif type(value) == bool:
            rdplot_call += ''.join([',',key,'=',str(value).upper()])
        elif type(value) == type(None):
            rdplot_call += ''.join([',',key,'=NULL'])
        else:
            rdplot_call += ''.join([',',str(key),'=',str(value)])

    function_call = '\n'.join(filter(None,
                                     ['library(rdrobust)',
                                      "df = read.csv('temp_file_for_rddesign.csv')",
                                      "out = rdplot(%s)" %rdplot_call]))

    # Modifications to the dataframe to get rid of infinite values
    df = df.loc[:,varlst].replace([np.inf, -np.inf], np.nan)

    # here we print to csv, then run the R call, then remove the csv
    df.to_csv('temp_file_for_rddesign.csv')
    out = ro.r(function_call)
    if verbose:
        if covs:
            print('RD of %s on %s with covariates: %s' %(y,x,', '.join(covs)))
        else:
            print('RD of %s on %s' %(y,x))
        if dropped_vars:
            print('Dropped due to singularity:',', '.join(dropped_vars))
        print(out)
    os.remove('temp_file_for_rddesign.csv')

    elements = dict()
    for t in [x for x in out.names if x!='rdplot']:
        if t=='vars_poly' or t=='vars_bins':
            temp_elements = dict()
            for s in out.rx2(t).names:
                temp_elements[s] = ro.pandas2ri.ri2py(out.rx2(t).rx2(s))
            elements[t] = pd.DataFrame(temp_elements)
        else:
            try:
                elements[t] = ro.pandas2ri.ri2py(out.rx2(t))
            except:
                try:
                    elements[t] = out.rx2(t)[0]
                except:
                    print('Failed to pull',t)
                    pass

    line_output = elements['vars_poly']
    bin_output = elements['vars_bins']
    bin_output = bin_output.rename(columns={'rdplot_N':'Obs'})

    if hide:
        result = rd_dict(text_rdplot_arg=rdplot_call,**elements)
    else:
        fig = plt.figure()
        if size:
            ax = sns.scatterplot(x='rdplot_mean_x',y='rdplot_mean_y',
                                 data=bin_output,s=75,size='Obs',legend=legend)
        else:
            ax = sns.scatterplot(x='rdplot_mean_x',y='rdplot_mean_y',
                                 data=bin_output,s=75)
        if ci:
            plt.errorbar(bin_output.rdplot_mean_x,bin_output.rdplot_mean_y,
                         yerr=(bin_output.rdplot_ci_l-bin_output.rdplot_ci_r)/2,
                               fmt='none',capsize=2,elinewidth=1)

        plt.axvline(c,color=sns.color_palette()[1],linewidth=1,linestyle='--')
        sns.lineplot(x=line_output.rdplot_x[line_output.rdplot_x<c],
                     y=line_output.rdplot_y[line_output.rdplot_x<c],
                     ax=ax,color=sns.color_palette()[0],linewidth=2)
        sns.lineplot(x=line_output.rdplot_x[line_output.rdplot_x>c],
                     y=line_output.rdplot_y[line_output.rdplot_x>c],
                     ax=ax,color=sns.color_palette()[0],linewidth=2)
        ax.set(xlabel=x,ylabel=y)
        # Result includes the plot axis
        result = rd_dict(fig=fig,ax=ax,text_rdplot_arg=rdplot_call,**elements)

    return result


###############################################################################
###############################################################################
### Robust RD Estimation
###############################################################################
###############################################################################
def rdrobust(y, x, df, covs = None, dummies = None, c = 0, fuzzy = None, deriv = 0,
             p = 1, q=2, h = None, bwselect = 'mserd', vce = 'nn', cluster = None,
             nnmatch = 3, level = 95, b = None, rho = None, kernel = 'tri',
             weights = None, masspoints = 'adjust', bwcheck = None,
             bwrestrict = True, stdvars = False, scalepar = 1, scaleregul = 1,
             sharpbw = False, rep_all = True, subset = None, verbose=True,
             covs_drop = True):

    '''
    Implements local polynomial Regression Discontinuity (RD) point estimators with
    robust bias-corrected confidence intervals and inference procedures. See here:
    https://www.rdocumentation.org/packages/rdrobust/versions/0.99.4/topics/rdrobust.

    Parameters
    ----------
    y: is the dependent variable.

    x: is the running variable (a.k.a. score or forcing variable).

    covs: specifies additional covariates to be used in polynomial regression.
    They should be a list of columns names in your dataframe (strings).

    dummies: specifies dummy variable covariates to be used in polynomial regression.
    They should be a list of columns names in your dataframe (strings).
    Will automatically generate binary variables for each droping the first.

    c: specifies the RD cutoff in x; default is c = 0.

    verbose: (if True) displays the underlying R summary o the RD results.

    fuzzy: specifies the treatment status variable used to implement fuzzy RD
    estimation (or Fuzzy Kink RD if deriv=1 is also specified). Default is
    Sharp RD design and hence this option is not used.

    deriv: specifies the order of the derivative of the regression functions
    to be estimated. Default is deriv=0 (for Sharp RD, or for Fuzzy RD if fuzzy
    is also specified). Setting deriv=1 results in estimation of a Kink RD design
    (up to scale), or Fuzzy Kink RD if fuzzy is also specified.

    p: specifies the order of the local-polynomial used in the point-estimator;
    default is p = 1 (local linear regression).

    q: specifies the order of the local-polynomial used in the bias-correction;
    default is q = 2 (local quadratic regression).

    h: specifies the main bandwidth used to construct the RD point estimator.
    If not specified, bandwidth h is computed by the companion command rdbwselect.
    If two bandwidths are specified, the first bandwidth is used for the data below
    the cutoff and the second bandwidth is used for the data above the cutoff.

    b: specifies the bias bandwidth used to construct the bias-correction estimator.
    If not specified, bandwidth b is computed by the companion command rdbwselect.
    If two bandwidths are specified, the first bandwidth is used for the data below
    the cutoff and the second bandwidth is used for the data above the cutoff.

    rho: specifies the value of rho, so that the bias bandwidth b equals h/rho.
    Default is rho = 1 if h is specified but b is not.

    kernel: is the kernel function used to construct local-polynomial estimator(s).
    Options are triangular (default option), epanechnikov and uniform.

    weights: the variable used for optional weighting of the estimation procedure.
    The unit-specific weights multiply the kernel function.

    bwselect: specifies the bandwidth selection procedure to be used. By default
    it computes both h and b, unless rho is specified, in which case it only
    computes h and sets b=h/rho. Options are:

        mserd one common MSE-optimal bandwidth selector for the RD
        treatment effect estimator.

        msetwo two different MSE-optimal bandwidth selectors
        (below and above the cutoff) for the RD treatment effect estimator.

        msesum one common MSE-optimal bandwidth selector for the sum of
        regression estimates (as opposed to difference thereof).

        msecomb1 for min(mserd,msesum).

        msecomb2 for median(msetwo,mserd,msesum), for each side of the
        cutoff separately.

        cerrd one common CER-optimal bandwidth selector for the RD
        treatment effect estimator.

        certwo two different CER-optimal bandwidth selectors
        (below and above the cutoff) for the RD treatment effect estimator.

        cersum one common CER-optimal bandwidth selector for the sum of regression
        estimates (as opposed to difference thereof).

        cercomb1 for min(cerrd,cersum).

        cercomb2 for median(certwo,cerrd,cersum), for each side of the cutoff
        separately.

        Note: MSE = Mean Square Error; CER = Coverage Error Rate. Default is
        bwselect=mserd. For details on implementation see Calonico, Cattaneo and
        Titiunik (2014a), Calonico, Cattaneo and Farrell (2018), and Calonico,
        Cattaneo, Farrell and Titiunik (2017), and the companion software articles.

    masspoints: checks/controls for mass points in the running variable. Options:

        (i) "off" ignores the presence of mass points

        (ii) "check" looks for and reports the number of unique observations
        at each sideof the cutoff

        (iii) "adjust" controls that the preliminary bandwidths used in the
        calculations contain a minimal number of unique observations.
        By default it uses 10 observations, but it can be manually adjusted
        with the option bwcheck).

    bwcheck: if a positive integer is provided, the preliminary bandwidth used in the
    calculations is enlarged so that at least bwcheck unique observations are used

    bwrestrict: if True, computed bandwidths are restricted to lie within the
    range of x; default is bwrestrict = True.

    stdvars: if True, x and y are standardized before computing the bandwidths;
    default is stdvars = False.

    vce: specifies the procedure used to compute the variance-covariance matrix
    estimator (Default is vce=nn). Options are

        nn for heteroskedasticity-robust nearest neighbor variance estimator with
        nnmatch the (minimum) number of neighbors to be used.

        hc0 for heteroskedasticity-robust plug-in residuals variance
        estimator without weights.

        hc1 for heteroskedasticity-robust plug-in residuals variance
        estimator with hc1 weights.

        hc2 for heteroskedasticity-robust plug-in residuals variance
        estimator with hc2 weights.

        hc3 for heteroskedasticity-robust plug-in residuals variance
        estimator with hc3 weights.

    cluster: indicates the cluster ID variable used for cluster-robust variance
    estimation with degrees-of-freedom weights. By default it is combined with
    vce=nn for cluster-robust nearest neighbor variance estimation. Another
    option is plug-in residuals combined with vce=hc0.

    nnmatch: to be combined with for vce=nn for heteroskedasticity-robust nearest
    neighbor variance estimator with nnmatch indicating the minimum number of
    neighbors to be used. Default is nnmatch=3

    level: sets the confidence level for confidence intervals; default is level = 95.

    scalepar: specifies scaling factor for RD parameter of interest. This option
    is useful when the population parameter of interest involves a known
    multiplicative factor (e.g., sharp kink RD). Default is scalepar = 1
    (no scaling).

    scaleregul: specifies scaling factor for the regularization term added to the
    denominator of the bandwidth selectors. Setting scaleregul = 0 removes the
    regularization term from the bandwidth selectors; default is scaleregul = 1.

    sharpbw: option to perform fuzzy RD estimation using a bandwidth selection
    procedure for the sharp RD model. This option is automatically selected if
    there is perfect compliance at either side of the cutoff.

    rep_all: if specified, rdrobust reports three different procedures

        conventional RD estimates with conventional standard errors.

        bias-corrected estimates with conventional standard errors.

        bias-corrected estimates with robust standard errors.

    subset: an optional vector specifying a subset of observations to be used.

    covs_drop: In python, automatically drops any covariates that take on
    ONLY one value on the front end (fixing issue with original rdplot code)
    and also activates covs_drop in R which checks for other multicollinearity.

    Returns
    -------
    N: vector with the sample sizes used to the left and to the right of the cutoff.

    N_h: vector with the effective sample sizes used to the left and to
    the right of the c cutoff value.

    p: order of the polynomial used for estimation of the regression function.

    q: order of the polynomial used for estimation of the bias of the regression.

    bws: matrix containing the bandwidths used.

    tau_cl: conventional local-polynomial estimate to the left and to the
    right of the cutoff.

    tau_bc: bias-corrected local-polynomial estimate to the left and to the
    right of the cutoff.

    coef: vector of conventional and bias-corrected local-polynomial RD estimates.

    se: vector containing conventional and robust standard errors of the
    local-polynomial RD estimates.

    bias: estimated bias for the local-polynomial RD estimator below/above cutoff.

    beta_p_l: conventional p-order local-polynomial estimates left of the cutoff.

    beta_p_r: conventional p-order local-polynomial estimates right of the cutoff.

    V_cl_l: conventional variance-covariance matrix estimated below the cutoff.

    V_cl_r: conventional variance-covariance matrix estimated above the cutoff.

    V_rb_l: robust variance-covariance matrix estimated below the cutoff.

    V_rb_r: robust variance-covariance matrix estimated above the cutoff.

    pv vector: containing the p-values associated with conventional,
    bias-corrected and robust local-polynomial RD estimates.

    ci: matrix containing the confidence intervals associated with conventional,
    bias-corrected and robust local-polynomial RD estimates.
    '''
    # Pull in the dictionary of all the available variables
    d = vars()

    # Helper Classes (Bad Programming)
    class Capturing(list):
        def __enter__(self):
            self._stdout = sys.stdout
            sys.stdout = self._stringio = StringIO()
            return self
        def __exit__(self, *args):
            self.extend(self._stringio.getvalue().splitlines())
            del self._stringio    # free up some memory
            sys.stdout = self._stdout
    class rd_dict:
        # So we get nice output._____ entries, this is bad coding
        def __init__(self, **entries):
            self.__dict__.update(entries)


    # At the very least, cut out missing values of the two main variables
    varlst = [x,y]
    df = df.dropna(subset=varlst).copy()
    dropped_vars = []

    # General all purpose roptions
    roptions = ['c','deriv','p','q','h','bwselect','vce','nnmatch','level','b',
                'rho','kernel','weights','scalepar','scaleregul','sharpbw',
                'masspoints','bwcheck','bwrestrict','covs_drop','stdvars']

    if covs:
        # convert singular covariates into a list for simplicity
        if type(covs)==str:
            covs = [covs]
        elif type(covs)==tuple:
            covs = list(covs)
        # dealing with smart removal
        if covs_drop:
            tempdf = df.loc[:,varlst].copy()
            for ii in covs:
                tempdf = tempdf.join(df[ii])
                if len(tempdf.dropna()[ii].unique())==1:
                    covs = [x for x in covs if x!=ii]
                    dropped_vars += [ii]
                    tempdf = tempdf.drop(columns=ii)
                tempdf = tempdf.dropna()
        # append the result regardless
        varlst += covs

    df = df.dropna(subset=varlst).copy()

    if dummies:
        if type(dummies)==str:
            dummies = [dummies]

        for ii in dummies:
            tempdums = pd.get_dummies(df[ii],prefix='rddum_%s' %ii,drop_first=True)
            if tempdums.shape[1]>0:
                df = df.join(tempdums)
                if not covs:
                    covs = list(tempdums.columns)
                else:
                    covs = covs + list(tempdums.columns)
                varlst += list(tempdums.columns)

    # y and x are called every time
    rdplot_call = 'y=df$%s,' %y + 'x=df$%s' %x

    # Other variables are actually in the dataframe and need to be added to the call
    if covs:
        rdplot_call += ',covs=cbind(' + ','.join(['df$%s' %x for x in covs])+')'

    # Take a subset
    if subset:
        rdplot_call += ','+'subset='+'df$%s' %subset
        varlst += [subset]
    if fuzzy:
        rdplot_call += ','+'fuzzy='+'df$%s' %fuzzy
        varlst += [fuzzy]
    if cluster:
        rdplot_call += ','+'cluster='+'df$%s' %cluster
        varlst += [cluster]
    if rep_all:
        # all was changed to rep_all so as not to conflict with pythonic syntax
        rdplot_call += ','+'all=TRUE'

    # The rest are options to be added as needed depending on the
    # type, they are translated to R
    for key in roptions:
        value = d[key]
        if type(value) == str:
            rdplot_call += ''.join([',',key,'=',"'%s'" %value])
        elif type(value) == list:
            rdplot_call += ''.join([',',key,'=c(',''.join(
                [str(x)+',' for x in value])[:-1],')'])
        elif type(value) == bool:
            rdplot_call += ''.join([',',key,'=',str(value).upper()])
        elif type(value) == type(None):
            rdplot_call += ''.join([',',key,'=NULL'])
        else:
            rdplot_call += ''.join([',',str(key),'=',str(value)])

    function_call = '\n'.join(filter(None,
                                     ['library(rdrobust)',
                                      "df = read.csv('temp_file_for_rddesign.csv')",
                                      "out = rdrobust(%s)" %rdplot_call]))

    # Modifications to the dataframe to get rid of infinite values
    df = df.loc[:,varlst].replace([np.inf, -np.inf], np.nan)

    # output, execute, delete
    df.to_csv('temp_file_for_rddesign.csv')
    with Capturing() as output:
        out = ro.r(function_call)
    os.remove('temp_file_for_rddesign.csv')
    masspointsprint = [i for i,x in enumerate(output) if 'Mass points' in x]

    elements = dict()
    for t in [x for x in out.names if x!='all']:
        try:
            elements[t] = ro.pandas2ri.ri2py(out.rx2(t))
        except:
            try:
                elements[t] = out.rx2(t)[0]
            except:
                print('Failed to pull',t)
                pass

    result = rd_dict(text_rdplot_arg=rdplot_call,**elements)
    printout = pd.DataFrame(np.concatenate(
        (result.coef,result.se,result.z,result.pv),axis=1),
                            columns=['Coef.','Std. Error','z','p>z']).join(
                                    pd.DataFrame(result.ci,
                                                 columns=['95% Lower','95% Upper']))
    printout.index = ['Conventional','Bias-Corrected','Robust']
    result = rd_dict(text_rdplot_arg=rdplot_call,printout=printout,**elements)

    if verbose:
        print(out)
        if covs:
            print('RD of %s on %s with covariates: %s' %(y,x,', '.join(covs)))
            if dropped_vars:
                print('Dropped due to singularity:',', '.join(dropped_vars))
        else:
            print('RD of %s on %s' %(y,x))
        if masspointsprint and masspoints=='check':
            print('Mass points detected in the running variable (no adjustment).')
        if masspointsprint and masspoints=='adjust':
            print('Mass points detected in the running variable (adjustment made).')
        print('')
        print(np.round(printout,3))

    return result

###############################################################################
###############################################################################
### Bandwidth Selection for RD Design
###############################################################################
###############################################################################
def rdbwselect(y, x, df, covs=None, dummies = None, c = 0, fuzzy = None, deriv = 0,
               p = 1, q=2, bwselect = 'mserd', vce = 'nn', cluster = None,
               nnmatch = 3,kernel = 'tri', weights = None, scaleregul = 1,
               sharpbw = False, masspoints = 'adjust', bwcheck = None,
               bwrestrict = True, stdvars = False, rep_all = False, subset = None,
               verbose=True, covs_drop = True):

    '''
    Implements bandwidth selectors for local polynomial Regression Discontinuity (RD)
    point estimators and inference procedures. See here:
    https://www.rdocumentation.org/packages/rdrobust/versions/0.99.4/topics/rdbwselect.

    Parameters
    ----------
    y: is the dependent variable.

    x: is the running variable (a.k.a. score or forcing variable).

    covs: specifies additional covariates to be used in the polynomial regression.
    They should be a list of columns names in your dataframe (strings).

    dummies: specifies dummy variable covariates to be used in polynomial regression.
    They should be a list of columns names in your dataframe (strings).
    The program will automatically generate binary variables for each and drops first.

    c: specifies the RD cutoff in x; default is c = 0.

    verbose: (if True) displays the underlying R summary o the RD results.

    fuzzy: specifies the treatment status variable used to implement fuzzy RD
    estimation (or Fuzzy Kink RD if deriv=1 is also specified). Default is Sharp RD
    design and hence this option is not used.

    deriv: specifies the order of the derivative of the regression functions to be
    estimated. Default is deriv=0 (for Sharp RD, or for Fuzzy RD if fuzzy is also
    specified). Setting deriv=1 results in estimation of a Kink RD design
    (up to scale), or Fuzzy Kink RD if fuzzy is also specified.

    p: specifies the order of the local-polynomial used to construct the
    point-estimator; default is p = 1 (local linear regression).

    q: specifies the order of the local-polynomial used to construct the
    bias-correction; default is q = 2 (local quadratic regression).

    kernel: is the kernel function used to construct the local-polynomial
    estimator(s). Options are triangular (default option), epanechnikov and uniform.

    weights: is the variable used for optional weighting of the estimation procedure.
    The unit-specific weights multiply the kernel function.

    bwselect: specifies the bandwidth selection procedure to be used. Options are

        mserd one common MSE-optimal bandwidth selector for the RD
        treatment effect estimator.

        msetwo two different MSE-optimal bandwidth selectors
        (below and above the cutoff) for the RD treatment effect estimator.

        msesum one common MSE-optimal bandwidth selector for the sum of
        regression estimates (as opposed to difference thereof).

        msecomb1 for min(mserd,msesum).

        msecomb2 for median(msetwo,mserd,msesum), for each side of the
        cutoff separately.

        cerrd one common CER-optimal bandwidth selector for the RD
        treatment effect estimator.

        certwo two different CER-optimal bandwidth selectors
        (below and above the cutoff) for the RD treatment effect estimator.

        cersum one common CER-optimal bandwidth selector for the sum of regression
        estimates (as opposed to difference thereof).

        cercomb1 for min(cerrd,cersum).

        cercomb2 for median(certwo,cerrd,cersum), for each side of the cutoff
        separately.

        Note: MSE = Mean Square Error; CER = Coverage Error Rate. Default is
        bwselect=mserd. For details on implementation see Calonico, Cattaneo and
        Titiunik (2014a), Calonico, Cattaneo and Farrell (2018), and Calonico,
        Cattaneo, Farrell and Titiunik (2017), and the companion software articles.

    vce: specifies the procedure used to compute the variance-covariance
    matrix estimator (Default is vce=nn). Options are

        nn for heteroskedasticity-robust nearest neighbor variance estimator
        with nnmatch the (minimum) number of neighbors to be used.

        hc0 for heteroskedasticity-robust plug-in residuals variance estimator
        without weights.

        hc1 for heteroskedasticity-robust plug-in residuals variance estimator
        with hc1 weights.

        hc2 for heteroskedasticity-robust plug-in residuals variance estimator
        with hc2 weights.

        hc3 for heteroskedasticity-robust plug-in residuals variance estimator
        with hc3 weights.

    cluster: indicates the cluster ID variable used for cluster-robust variance
    estimation with degrees-of-freedom weights. By default it is combined with
    vce=nn for cluster-robust nearest neighbor variance estimation. Another option
    is plug-in residuals combined with vce=hc0.

    nnmatch: to be combined with for vce=nn for heteroskedasticity-robust nearest
    neighbor variance estimator with nnmatch indicating the minimum number of
    neighbors to be used. Default is nnmatch=3

    scaleregul: specifies scaling factor for the regularization term added to the
    denominator of the bandwidth selectors. Setting scaleregul = 0 removes the
    regularization term from the bandwidth selectors; default is scaleregul = 1.

    sharpbw: option to perform fuzzy RD estimation using a bandwidth selection
    procedure for the sharp RD model. This option is automatically selected if
    there is perfect compliance at either side of the threshold.

    rep_all: if specified, rdbwselect reports all available bandwidth
    selection procedures.

    masspoints: checks/controls for mass points in the running variable. Options:

        (i) "off" ignores the presence of mass points

        (ii) "check" looks for and reports the number of unique observations
        at each sideof the cutoff

        (iii) "adjust" controls that the preliminary bandwidths used in the
        calculations contain a minimal number of unique observations.
        By default it uses 10 observations, but it can be manually adjusted
        with the option bwcheck).

    bwcheck: if a positive integer is provided, the preliminary bandwidth used in the
    calculations is enlarged so that at least bwcheck unique observations are used

    bwrestrict: if True, computed bandwidths are restricted to lie within the
    range of x; default is bwrestrict = True.

    stdvars: if True, x and y are standardized before computing the bandwidths;
    default is stdvars = False.

    subset: an optional vector specifying a subset of observations to be used.

    covs_drop: In python, automatically drops any covariates that take on
    ONLY one value on the front end (fixing issue with original rdplot code)
    and also activates covs_drop in R which checks for other multicollinearity.

    Returns
    -------
    N: vector with sample sizes to the left and to the righst of the cutoff.

    c: cutoff value.

    p: order of the local-polynomial used to construct the point-estimator.

    q: order of the local-polynomial used to construct the bias-correction estimator.

    bws: matrix containing the estimated bandwidths for each selected procedure.

    bwselect: bandwidth selection procedure employed.

    kernel: kernel function used to construct the local-polynomial estimator(s).
   '''
    # Pull in the dictionary of all the available variables
    d = vars()

    # Helper Classes (Bad Programming)
    class Capturing(list):
        def __enter__(self):
            self._stdout = sys.stdout
            sys.stdout = self._stringio = StringIO()
            return self
        def __exit__(self, *args):
            self.extend(self._stringio.getvalue().splitlines())
            del self._stringio    # free up some memory
            sys.stdout = self._stdout
    class rd_dict:
        # So we get nice output._____ entries, this is bad coding
        def __init__(self, **entries):
            self.__dict__.update(entries)

    # At the very least, cut out missing values of the two main variables
    varlst = [x,y]
    df = df.dropna(subset=varlst).copy()
    dropped_vars = []

    # General all purpose roptions
    roptions = ['c','deriv','p','q','bwselect','vce','nnmatch','kernel',
                'weights','scaleregul','sharpbw','masspoints','bwcheck',
                'bwrestrict','covs_drop','stdvars']

    if covs:
        # convert singular covariates into a list for simplicity
        if type(covs)==str:
            covs = [covs]
        elif type(covs)==tuple:
            covs = list(covs)
        # dealing with smart removal
        if covs_drop:
            tempdf = df.loc[:,varlst].copy()
            for ii in covs:
                tempdf = tempdf.join(df[ii])
                if len(tempdf.dropna()[ii].unique())==1:
                    covs = [x for x in covs if x!=ii]
                    dropped_vars += [ii]
                    tempdf = tempdf.drop(columns=ii)
                tempdf = tempdf.dropna()
        # append the result regardless
        varlst += covs

    df = df.dropna(subset=varlst).copy()

    if dummies:
        if type(dummies)==str:
            dummies = [dummies]
        for ii in dummies:
            tempdums = pd.get_dummies(df[ii],prefix='rddum_%s' %ii,drop_first=True)
            if tempdums.shape[1]>0:
                df = df.join(tempdums)
                if not covs:
                    covs = list(tempdums.columns)
                else:
                    covs = covs + list(tempdums.columns)
                varlst += list(tempdums.columns)

    # y and x are called every time
    rdplot_call = 'y=df$%s,' %y + 'x=df$%s' %x

    # Other variables are actually in the dataframe and need to be added to the call
    if covs:
        rdplot_call += ',covs=cbind(' + ','.join(['df$%s' %x for x in covs])+')'

    # Take a subset
    if subset:
        rdplot_call += ','+'subset='+'df$%s' %subset
        varlst += [subset]
    if fuzzy:
        rdplot_call += ','+'fuzzy='+'df$%s' %fuzzy
        varlst += [fuzzy]
    if cluster:
        rdplot_call += ','+'cluster='+'df$%s' %cluster
        varlst += [cluster]
    if rep_all:
        # all was changed to rep_all so as not to conflict with pythonic syntax
        rdplot_call += ','+'all=TRUE'

    # The rest are options to be added as needed depending on the type,
    # they are translated to R
    for key in roptions:
        value = d[key]
        if type(value) == str:
            rdplot_call += ''.join([',',key,'=',"'%s'" %value])
        elif type(value) == list:
            rdplot_call += ''.join([',',key,'=c(',''.join(
                [str(x)+',' for x in value])[:-1],')'])
        elif type(value) == bool:
            rdplot_call += ''.join([',',key,'=',str(value).upper()])
        elif type(value) == type(None):
            rdplot_call += ''.join([',',key,'=NULL'])
        else:
            rdplot_call += ''.join([',',str(key),'=',str(value)])

    function_call = '\n'.join(filter(None,
                                     ['library(rdrobust)',
                                      "df = read.csv('temp_file_for_rddesign.csv')",
                                      "out = rdbwselect(%s)" %rdplot_call]))


    # Modifications to the dataframe to get rid of infinite values
    df = df.loc[:,varlst].replace([np.inf, -np.inf], np.nan)

    # run the thing
    df.to_csv('temp_file_for_rddesign.csv')
    with Capturing() as output:
        out = ro.r(function_call)
    os.remove('temp_file_for_rddesign.csv')
    masspointsprint = [i for i,x in enumerate(output) if 'Mass points' in x]

    elements = dict()
    for t in [x for x in out.names if x!='all']:
        try:
            elements[t] = ro.pandas2ri.ri2py(out.rx2(t))
        except:
            try:
                elements[t] = out.rx2(t)[0]
            except:
                print('Failed to pull',t)
                pass

    if type(elements['bw_list'])!=list:
        elements['bw_list'] = [elements['bw_list']]
    result = rd_dict(text_rdplot_arg=rdplot_call,**elements)
    printout = pd.DataFrame(data=result.bws,index=result.bw_list,
                            columns=['BW est. (h) Left','BW est. (h) R',
                                     'BW bias. (b) Left','BW bias. (b) Right'])
    result = rd_dict(text_rdplot_arg=rdplot_call,printout=printout,**elements)

    if verbose:
        print(out)
        if covs:
            print('RD of %s on %s with covariates: %s' %(y,x,', '.join(covs)))
            if dropped_vars:
                print('Dropped due to singularity:',', '.join(dropped_vars))
        else:
            print('RD of %s on %s' %(y,x))
        if masspointsprint and masspoints=='check':
            print('Mass points detected in the running variable (no adjustment).')
        if masspointsprint and masspoints=='adjust':
            print('Mass points detected in the running variable (adjustment made).')
        print('')
        print(np.round(printout,3))

    return result
