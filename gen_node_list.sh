#!/usr/bin/env bash

set -euo pipefail

source "config.sh"

N=120

#ssh $RH /u/ccc/bin/cdfruptime > /tmp/cdf-hosts

cat /tmp/cdf-hosts |
    grep -v 2200 | grep -v 2210 |
    sed -r 's/up.*([0-9]+) users?,\s/\1/' |                     # Make the rows consistent in terms of column alignment
    sort -k 3 |                                                 # Sort by number of logged in users
    awk '{ print($1 ".teach.cs.toronto.edu") }' |               # Convert host table to proper names
    head -n "$(( N + 1 ))" > cmake-build-debug/cdf_machines     # Dump N hosts + one 'root' dude to run MPI from

echo "Dudes:"
cat cmake-build-debug/cdf_machines

echo "DONE!"
