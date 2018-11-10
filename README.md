# UofT CSC2306F: High-Performance Scientific Computing Assignments

## Overview

These assignments consist mostly of MPI code implemented in C++, plus some Python used to analyze the experimental
results and create the plots.

## Getting Started

The code requires Open MPI and CMake (3.5, so old CMakes are OK. The most recent is 3.13.) to be installed, as well 
as a C++11-compatible compiler (the g++ on CDF is OK).


### Dependencies

 - `Open MPI` for the bread and butter multiprocessing.
 - `gflags`  for clean declarative argument parsing.
```bash
git clone https://github.com/gflags/gflags && cd gflags && mkdir build
cd build && cmake -DEXPORT_BUILD_DIR=ON .. && make -j4
```
 - (Optional) Python 3 for analyzing the data and producing the plots. The Python package dependencies are 
 specified in the `requirements.txt` file, which can be loaded easily into any virtual or Anaconda environment.
 
 
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
Modigy `PROBLEM` in the preamble of `a01.sh` to choose what problem to run (2 or 4).
    
 

## Project Structure

The `src` directory contains all the useful source code grouped by assignment (1--3), with a `common` directory 
containing some shared utilities. The source code includes the actual C++ assignment code, plus the analysis 

The `results` directory will contain experiment results.

The `config` directory contains configuration files such as (potentially dynamically generated) lists of machines to 
run the MPI code on.
