#@IgnoreInspection BashAddShebang

CMAKE_DIR="cmake-build-debug"
CDF_USER="barsanio"
REMOTE_HOST="wolf.cdf.toronto.edu"
# Please use an SSH keypair to connect. Or paste in your password. I'm not your mom, you can do what you want.

# Remote host
RH="${CDF_USER}@${REMOTE_HOST}"
# Remote project dir
RPROJ="$RH:~/hpsc"

# Remote bin dir
#RBIN="$RH:~/hpsc/bin"
