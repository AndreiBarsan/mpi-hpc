#!/usr/bin/env bash

set -eu -o pipefail

CMAKE_DIR="cmake-build-debug"

if ! [[ -d "$CMAKE_DIR" ]]; then
    read -e -p "CMake was not run yet, it would seem. Run now? [y/n] "
    case "$REPLY" in
        [yY])
            echo "Ok, running."
            ;;
        [nN])
            echo "Aborting."
            exit 1
            ;;
        **)
            echo "Unknown input, aborting."
            exit 1
            ;;
    esac
    mkdir "$CMAKE_DIR" && cd $_ && cmake ..
fi

(cd $CMAKE_DIR && make -j$(gnproc))

N_NODES=4

# TODO make this into a makefile maybe
mpirun -np "$N_NODES" -machinefile config/local-machine.txt $CMAKE_DIR/globsum