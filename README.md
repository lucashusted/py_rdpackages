# State of the Art RD Packages For Python

## Installation

Please note that installation should be done with a mixture of pip and direct installations in R so that the correct versions of packages are installed.

With pip for python:
`pip install py_rdpackages`

Important: you need to have the original `rdrobust` installed in R:
`install.packages('rdrobust')`

Notes:
- `rdrobust` requires the newest version of `ggplot2` so please: `install.packages('ggplot2')`
- The `rdrobust` and `ggplot2` versions in Anaconda lag behind the current release for R. Please install in R directly as opposed to through conda.


## Introduction and Use
These packages are a work in progress, but are an attempt to create a wrapper to implement the wonderful RD packages found here (https://sites.google.com/site/rdpackages/rdrobust) which utilize R or Stata, so that they can be used in Python directly.

I utilize the graphics package of Seaborn to make the plots look nice and to enable additional features, like binning points by size.

There are three packages in `py_rdpackages`:
1. `rdplot` creates plots of the regression discontinuity with a variety of options.
2. `rdrobust` does the RD and reports the regression results.
3. `rdbwselect` selects the optimal bandwidth size.

## A Example of the Output
The code in the testing folder produces the following example.

![Alt text](/testing/sample_rdplot.png?raw=true "Regression Discontinuity")

## Requirements and Stability

Use of the programs requires all of the following packages in Python:
1. `rpy2` for running R in Python
2. `matplotlib` and `seaborn` for producing high quality graphics
3. `pandas`, `numpy` and for data manipulation and dataframe reading

Currently tested and stable for:
- `rdrobust` version 0.99.9
- `rpy2` version 2.9.4 (last version updated through conda -- may work for later versions)
- `ggplot2` version 3.3.0


## Limitations
1. `ryp2` produced slow pandas DF to R DF conversions, so I use `pd.df.to_csv('temp_file_for_rd.csv')` as a solution and then delete that same file after doing the analysis. This should be fixed in future versions.
2. Of course, Python calling R and then converting back to python is not ideal. Some future version should code this from scratch.
