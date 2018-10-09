#!/usr/bin/env bash
# Builds the code (tailored for CDF, should unify with .sh.inc).

set -euo pipefail

# for testing (very slow on CDF...)
#rm -rf hpsc/build

if [[ ! -d ~/hpsc/build ]]; then
    echo -e "\nNo build dir; it seems CMake hasn't been run. Running now.\n\n"
    mkdir ~/hpsc/build
    cd ~/hpsc/build
    cmake ..
fi

cd ~/hpsc/build
make -j$(nproc)

mkdir -p ~/hpsc/build/results

