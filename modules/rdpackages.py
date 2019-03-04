'''
This code was conceived as a general wrapper to do RD Estimation in Python based on the wonderful codes of Cattaneo et al (https://www.rdocumentation.org/packages/rdrobust/versions/0.99.4/topics/rdplot). You should reference their documentation for all technical details and let me know if there are any errors.
'''
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

###############################################################################
###############################################################################
### Plotting of RD Design
###############################################################################
###############################################################################
def rdplot(y,x,df,covs=None,x_range=[],
           c = 0, p = 4, nbins = None, binselect = 'esmv',
           scale = None, kernel = 'uni', weights = None, h = None,
           support = None,subset=None,hide=False,R_options='',verbose=False,size=True,legend=False):
    '''
    This function is adapted from rdplot by Cattaneo et al. Options have been mapped to python datatypes:

    y  is the dependent variable. It should be a string representing a column in your dataframe.

    x  is the running variable (a.k.a. score or forcing variable). It should also be a string column.

    df specifies the pandas dataframe where this data is coming from

    covs specifies additional covariates to be used in the polynomial regression. They should be a list of columns names in your dataframe (strings).

    x_range allows you to trim the dataframe. This is seperate from the R option "support" as it is crude and will just remove all observations outside the range of the running variable.

    c  specifies the RD cutoff in x; default is c = 0.

    p  specifies the order of the global-polynomial used to approximate the population conditional mean functions for control and treated units; default is p = 4.

    nbins  specifies the number of bins used to the left of the cutoff, denoted J−, and to the right of the cutoff, denoted J+, respectively. If not specified, J+ and J− are estimated using the method and options chosen below.

    binselect specifies the procedure to select the number of bins. This option is available only if J− and J+ are not set manually. Options are:
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

    R_options allows you to manually insert additional options (see R documentation of the original program for more information -- https://www.rdocumentation.org/packages/rdrobust/versions/0.99.4/topics/rdplot). You should insert any options as a string, in the same format as you would in R. However, this should not really be used. Most of these options are for formatting R plots, which are default going to be turned off here.

    verbose has it print the rdplot call from R for you

    size (if true) means that your scatterplot will have dots that grow larger with the mass at that point.

    legend only matters if size is turned on. This should be either 'brief', 'full', or False

    '''
    # Pull in the dictionary of all the available variables
    d = vars()

    # General all purpose roptions
    roptions = ['c','p','nbins','binselect','scale','kernel','weights','h','support']

    # y and x are called every time
    rdplot_call = 'y=df$%s,' %y + 'x=df$%s,' %x + 'hide=TRUE'
    if covs:
        all_covars = ',covs=cbind(' + ','.join(['df$%s' %x for x in covs])
        rdplot_call += all_covars[:-1]+')'
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
        df = df[df[x].between(x_range[0],x_range[1])]
    df.to_csv('temp_file_for_rdplot.csv')
    out = ro.r(function_call)
    if verbose:
        print(out)
    os.remove('temp_file_for_rdplot.csv')

    elements = dict()
    for t in out.names:
        try:
            elements[t] = ro.pandas2ri.ri2py(out.rx2(t))
        except:
            print("There was an error, probably with importing R code")
            pass

    line_output = elements['vars_poly']
    bin_output = elements['vars_bins']
    bin_output.rename(columns={'rdplot_N':'Obs'},inplace=True)

    class rd_dict:
        # So we get nice output._____ entries, this is bad coding
        def __init__(self, **entries):
            self.__dict__.update(entries)

    if hide:
        result = rd_dict(text_rdplot_arg=rdplot_call,**elements)
    else:
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

        result = rd_dict(ax=ax,text_rdplot_arg=rdplot_call,**elements)

    return result


###############################################################################
###############################################################################
### Robust RD Estimation
###############################################################################
###############################################################################
def rdrobust(y, x, df, covs=[], x_range=[], c = 0, fuzzy = None, deriv = 0, p = 1, q=2, h = None,
             bwselect = 'mserd', vce = 'nn', cluster = None, nnmatch = 3, level = 95, b = None,
             rho = None, kernel = 'tri', weights = None, scalepar = 1, scaleregul = 1,
             sharpbw = False, rep_all = False, subset = None, verbose=True):

    # Pull in the dictionary of all the available variables
    d = vars()

    # General all purpose roptions
    roptions = ['c','deriv','p','q','h','bwselect','vce',
                'nnmatch','level','b','rho','kernel','weights','scalepar',
                'scaleregul','sharpbw']

    # y and x are called every time
    rdplot_call = 'y=df$%s,' %y + 'x=df$%s' %x

    # Other variables are actually in the dataframe and need to be added to the call
    if covs:
        all_covars = ',covs=cbind(' + ','.join(['df$%s' %x for x in covs])
        rdplot_call += all_covars[:-1]+')'
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
        print(np.round(printout,2))

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

    # Pull in the dictionary of all the available variables
    d = vars()

    # General all purpose roptions
    roptions = ['c','deriv','p','q','bwselect','vce',
                'nnmatch','kernel','weights',
                'scaleregul','sharpbw']

    # y and x are called every time
    rdplot_call = 'y=df$%s,' %y + 'x=df$%s' %x

    # Other variables are actually in the dataframe and need to be added to the call
    if covs:
        all_covars = ',covs=cbind(' + ','.join(['df$%s' %x for x in covs])
        rdplot_call += all_covars[:-1]+')'
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
        print(np.round(printout,2))

    return result
