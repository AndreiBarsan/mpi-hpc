#!/usr/bin/env bash
# Used to run experiments from Assignment 1 on the CDF.

################################################################################
# Preamble
################################################################################
set -eu -o pipefail

source "config.sh"
source "build_cmake.sh.inc"

PROBLEM="4"


################################################################################
# Setup and Orchestration
################################################################################
echo "Will rsync code to CDF..."
time rsync -rv --exclude 'cmake-build-debug' \
    --exclude 'results' --exclude '.git' --exclude '.idea' \
     . "${RPROJ}"
echo "OK"

START_NODE="$(cat cdf_machines | tail -n 1)"
RH="${CDF_USER}@${START_NODE}"

echo "Will SSH to $RH. If you can't connect, make sure you are on the UofT network, e.g., using the VPN!"
echo "You may have to manually connect first and run 'kinit' MPI can work."

ssh "$RH" "hpsc/build.sh"

# The sed tails everything except the last line.
# We use the first line as the starting node, and the rest as workers.
ssh "$RH" cat "hpsc/cdf_machines | sed '\$d' >| /tmp/machines"


################################################################################
# Main experiments
################################################################################
echo -e "\n\n\tRunning problem $PROBLEM!\n\n"
if [[ "$PROBLEM" == "2" ]]; then
    for N_NODES in 2 4 8 16; do
        echo -e "\n\n\tExperiment with ${N_NODES} nodes!\n\n"

        ssh "$RH" mpirun -np $N_NODES -pernode -machinefile /tmp/machines hpsc/build/globsum --out_dir=hpsc/results
        ssh "$RH" mpirun -np $N_NODES -pernode -machinefile /tmp/machines hpsc/build/globsum --out_dir=hpsc/results --multiple_ops
    done
else
    for N_NODES in 2 4 8 16; do
        echo -e "\n\n\tExperiment with ${N_NODES} nodes!\n\n"

        ssh "$RH" mpirun -np $N_NODES -pernode -machinefile /tmp/machines hpsc/build/globsum --out_dir=hpsc/results --iterations=10
    done
fi

echo "Grabbing results back for number crunching..."
time rsync -rv "$RH:hpsc/results/" "results/cdf/"

