#@IgnoreInspection BashAddShebang

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

if hash nproc 1>/dev/null 2>&1; then
    N_PROC=$(nproc)
else
    N_PROC=$(gnproc)
fi
echo "Using $N_PROC processors to build."
(cd $CMAKE_DIR && make -j${N_PROC})

mkdir -p results
