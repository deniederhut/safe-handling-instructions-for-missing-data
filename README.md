# Safe Handling Instructions for Missing Data

## About

This repository contains the code used to generate the experimental data, one run of experimental data, and the analysis code used in "Safe Handling Instructions for Missing Data", a paper to be presented at the 2018 meeting of the Python in Science Conferences (SciPy 2018).

## How to use this repository

1. Install the libraries at their pinned versions (found in requirements.txt) into a virtual environment.
2. Open the analysis notebook with `jupyter notebook analysis.ipynb`
3. The notebook is pre-populated with the output from the last time it was executed. You are free to modify the code in the notebook and re-run it to see how it changes the output of the analysis.

To re-run the experiment which generated the data, run `python make_data.py`. Use the `-h` flag to see experimental parameters that you can set. Be warned that, depending on which options you choose, the total runtime of the experiment may extend beyond several days.
