# State of the Art RD Packages For Python

These packages are a work in progress, but are an attempt to create a wrapper to implement the wonderful RD packages found here (https://sites.google.com/site/rdpackages/rdrobust) which utilize R or Stata, so that they can be used in Python.

Use of the programs requires all of the following packages in Python:
1. `rpy2` for running R in Python
2. `matplotlib` and `seaborn` for producing high quality graphics
3. `pandas`, `numpy` and `statsmodels` for data manipulation and dataframe reading

Important: you need to have the original `rdrobust` installed in R (you can find this in the above link).

Current limitations to be fixed:
1. `ryp2` produced slow pandas DF to R DF conversions, so I use `pd.df.to_csv('temp_file_for_rd.csv')` as a solution and then delete that same file after doing the analysis. This should be fixed in future versions.
2. I have not written a full set of graphics options for the RD plots. Future versions will allow you to use all of the classic `matplotlib` tools, and also turn on/off the vertical line and change the coloring of the scatterplot. Also, I do not incorporate standard errors on the scatterplot bins, which would be easy enough to add. In this version, if you want to do some of those things, you should do them after making a function call.
3. Of course, Python calling R and then converting back to python is not ideal. Some future version should get the original C implementation of the code and just work from there.

Hopefully I will make these better in the future, but for now let me know if you spot any bugs.
