#!/usr/bin/env bash

set -eu -o pipefail

source "config.sh"

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

#for N_NODES in 2 4 8 16 24; do
#    mpirun -np "$N_NODES" -machinefile config/local-machine.txt $CMAKE_DIR/globsum
#done

#N_NODES=4

rsync -r --info=progress2 "$CMAKE_DIR/" "${RBIN}"

START_NODE="$(cat cmake-build-debug/cdf_machines | tail -n 1)"
RH="${CDF_USER}@${START_NODE}"

echo "Will SSH to $RH. If you can't connect, make sure you are on the UofT network, e.g., using the VPN!"
echo "You may have to manually connect first before passwordless SSH or MPI can work."
ssh "$RH" cat "hpsc/bin/cdf_machines"
# The sed tails everything except the last line.
ssh "$RH" cat "hpsc/bin/cdf_machines | sed '\$d' >| /tmp/machines"

echo "Will do the mpirun now!"
ssh "$RH" mpirun -np 4 -pernode -machinefile /tmp/machines hpsc/bin/globsum


