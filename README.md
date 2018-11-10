# UofT CSC2306F: High-Performance Scientific Computing Assignments, Fall 2018

## Overview

These assignments consist mostly of MPI code implemented in C++, plus some Python used to analyze the experimental
results and create the plots.

## Getting Started

The code requires Open MPI and CMake (3.5, so old CMakes are OK. The most recent is 3.13.) to be installed, as well 
as a C++11-compatible compiler (the g++ on UofT's CDF is OK).


### Dependencies

 - `Open MPI` for the bread and butter for distributed multiprocessing.
 - `gflags`  for clean declarative argument parsing. (Nice way to specify flags for your program.)
```bash
git clone https://github.com/gflags/gflags && cd gflags && mkdir build
cd build && cmake -DEXPORT_BUILD_DIR=ON .. && make -j4
```
 - (Optional) Python 3 for analyzing the data and producing the plots. The Python package dependencies are 
 specified in the `requirements.txt` file, which can be loaded easily into any virtual or Anaconda environment.
 - (Optional) For Assignment 2, Eigen 3 can enable additional checks by comparing the results produced by the custom 
 solver with those produced by an industry-standard solver provided by Eigen.
 
 
### Running the Code

To build the code and run some of the Assignment 01 experiments, simply use the `a01-local.sh` script.
This runs MPI locally. For best results, use a computing cluster like U of T's CDF.

A possible sequence of actions could be:

```bash
./a01-02-local.sh                       # This builds the code if necessary and starts an experiment.
virtualenv ~/.venv/hpsc                 # Create a virtual environment for the Python packages.
source ~/.venv/hpsc/bin/activate        # Activate the virtual environment.
pip install -r requirements.txt         # Install the dependencies.
cd src/a01
python analysis.py                      # Run the analysis script (may need to modify what directory it reads from).
```

Use `a01-04-loca.sh` to run Problem 4 locally.

To run either Problem 2 or Problem 4 on CDF, first run `gen_node_list.sh` to find free machines, then `a01.sh`.
Modify `PROBLEM` in the preamble of `a01.sh` to choose what problem to run (2 or 4).
    
 

## Project Structure

The `src` directory contains all the useful source code grouped by assignment (1--3), with a `common` directory 
containing some shared utilities. The source code includes the actual C++ assignment code, plus the Python 
scripts used for analysis, plotting, and some other smaller exercises.

The `results` directory will contain experiment results.

The `config` directory contains configuration files such as (potentially dynamically generated) lists of machines to 
run the MPI code on.
