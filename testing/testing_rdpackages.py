#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from py_rdpackages import rdrobust, rdplot, rdbwselect
import pandas as pd

# loading in a standard RD dataframe with a made up outcome and running variable
df = pd.read_csv('testing_data.csv')

# running the actual estimation
output = rdrobust('outcome','score',df)

# getting the plot (feel free to change fig, ax in plot.fig, plot.ax)
plot = rdplot('outcome','score',df,p=1)

# bandwidths
bws = rdbwselect('outcome','score',df)