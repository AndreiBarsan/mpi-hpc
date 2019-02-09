# Distributed Solvers for Large Systems Using MPI

## Overview

These assignments consist mostly of MPI code implemented in C++, plus some
Python used to analyze the experimental results and create the plots. I wrote
this code for coursework during my PhD.

## Getting Started

The code requires Open MPI, CMake 3.5 (not super new), gflags, and Eigen. The
to be installed, as well as a C++14-compatible compiler
(the g++ on UofT's CDF is OK). Several C++14 features are used, such as
`make_shared<T[]>`, but slightly older compilers should still support it.


### Dependencies

 - `Open MPI` for the bread and butter for distributed multiprocessing.
 - `gflags` for clean declarative argument parsing.
```bash
git clone https://github.com/gflags/gflags && cd gflags && mkdir build
cd build && cmake -DEXPORT_BUILD_DIR=ON .. && make -j4
```
 - Eigen 3 is *required* for Assignment 3, but optional for the previous two!
   A reasonably new version can be installed on a Debian-like distribution by
   running `sudo apt install libeigen3-dev`. Note that Eigen3 is a header-only
   library, so it doesn't need to be compiled separately or linked, just included. 
 - (Optional) Python 3 for analyzing the data and producing the plots. The Python package dependencies are
 specified in the `requirements.txt` file, which can be loaded easily into any virtual or Anaconda environment.
 - (Optional) For Assignment 2, Eigen 3 can enable additional checks by comparing the results produced by the custom
 solver with those produced by an industry-standard solver provided by Eigen. 


### Running the Code

To build the code and run some of the Assignment 01 experiments, simply use the
`a01-local.sh` script.  This runs MPI locally. For best results, use
a computing cluster like U of T's CDF.

A possible sequence of actions could be:

```bash
./a01-02-local.sh                       # This builds the code if necessary and starts an experiment.
virtualenv ~/.venv/hpsc                 # Create a virtual environment for the Python packages.
source ~/.venv/hpsc/bin/activate        # Activate the virtual environment.
pip install -r requirements.txt         # Install the dependencies.
cd src/a01
python analysis.py                      # Run the analysis script (may need to modify what directory it reads from).
```

#### Assignment 1

Use `a01-04-local.sh` to run Problem 4 locally.

To run either Problem 2 or Problem 4 on CDF, first run `gen_node_list.sh` to find free machines, then `a01.sh`.
Modify `PROBLEM` in the preamble of `a01.sh` to choose what problem to run (2 or 4).


#### Assignment 3

Use `./a03-02-deboor.sh` to run the second problem and dump experiment results.
You system should have 16+ cores for this to work; otherwise, the script should
be modified to use a computing cluster, such as UofT's CDF.

Use `python src/a03/plot_output_2d.py results/spline_2d_output` to visualize
the interpolation results, after running the experiments using the scripts
above. Warning: may open many plot windows.

Use `python src/a03/plot_timing.py results/spline_2d_output` to generate the
timing plots from the report.



## Project Structure

The `src` directory contains all the useful source code grouped by assignment (1--3), with a `common` directory
containing some shared utilities. The source code includes the actual C++ assignment code, plus the Python
scripts used for analysis, plotting, and some other smaller exercises.

The `results` directory will contain experiment results.

The `config` directory contains configuration files such as (potentially dynamically generated) lists of machines to
run the MPI code on.


## A Few MPI Hints and Tips

 * Make keeping track of which node is outputting what by passing the
 `--tag-output` flag to `mpirun`, thereby ensuring that every output
 line is prefixed with the ID of the node. `--timestamp-output` is
 self-explanatory and a nice addition, especially if you're not already
 using a dedicated logging library like `glog`.
