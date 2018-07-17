# ATL11
Python code to produce the Land-Ice Along-Track H(t) product (ATL11), based on the current version of the ATL11 ATBD. The code reads ATL06 data from a set of files (one per track per cycle) and writes out an ATL11 data file.  The command-line interface is found in ATL06_to_ATL11.py.

Files provided:
  ATL06_data.py: Scripts to read ATL06 simulated data 
  ATL06_pair.py: Data structures holding pairs of ATL06 measurements
  ATL06_to_ATL11.py: Script to create ATL11 files from ATL06 inputs
  ATL11_data.py: ATL11 class definition, include code to read and write the data
  ATL11_misc.py: Helper structures and defaults
  ATL11_output_attrs.csv: File structure definition used by ATL11_data.py
  ATL11_plot.py: Interactive plot of ATL11 output
  ATL11_point.py: Class for fitting ATL11 data to ATL06 data, contains methods called in ATL06_to_ATL11.py
  RDE.py: Robust Dispersion Estimate, to estimate the spread of a distribution based on its percentiles
  geo_index.py: Scripts for generating a geographic index for (so far) ATL06 data.
  poly_ref_surf.py: Class used for fitting polynomial surfaces to geograpic data
  run_Greenland_sim: script to generate command line calls that run ATL06_to_ATL11 for simulated data
