# UofT CSC2306F: High-Performance Scientific Computing Assignments

## Overview

These assignments consist mostly of MPI code implemented in C++, plus some Python used to analyze the experimental
results and create the plots.

## Getting Started

The code requires Open MPI and CMake to be installed, as well as a C++11-compatible compiler.


### Dependencies

 - `Open MPI` for the bread and butter multiprocessing.
 - `gflags`  for clean declarative argument parsing.
 - (Optional) Python 3.5+ for analyzing the data and producing the plots. The Python package dependencies are 
 specified in the `requirements.txt` file, which can be loaded easily into any virtual or Anaconda environment.
 
 
### Running the Code

To build the code and run some of the Assignment 01 experiments, simply use the `a01-local.sh` script.
This runs MPI locally. For best results, use a computing cluster (unless your CPU is a beast ;).

A possible sequence of actions could be:

```bash
./a01-local.sh
virtualenv ~/.venv/hpsc
source ~/.venv/bin/activate
pip install -r requirements.txt
cd src/a01
python analysis.py
```
    
 

## Project Structure

The `src` directory contains all the useful source code grouped by assignment (1--3), with a `common` directory 
containing some shared utilities. The source code includes the actual C++ assignment code, plus the analysis 

The `results` directory will contain experiment results.

The `config` directory contains configuration files such as (potentially dynamically generated) lists of machines to 
run the MPI code on.
