#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
    printf '%s\n' 'This preparation script requires an Apple Silicon Mac.' >&2
    exit 1
fi

xcrun --find clang >/dev/null
command -v brew >/dev/null || {
    printf '%s\n' 'Install Homebrew before running this script.' >&2
    exit 1
}

omp_prefix=$(brew --prefix libomp)
if [ ! -f "$omp_prefix/include/omp.h" ]; then
    brew install libomp
    omp_prefix=$(brew --prefix libomp)
fi

jobs=$(sysctl -n hw.ncpu)
printf 'Preparing Kimi K3 for %s\n' "$(sysctl -n machdep.cpu.brand_string)"
make -j"$jobs" OMP_PREFIX="$omp_prefix" all
make -j"$jobs" OMP_PREFIX="$omp_prefix" test
printf '\nEngine ready: %s/bin/k3\n' "$PWD"
printf '%s\n' 'Model weights are separate; see docs/MACOS.md for storage and validation status.'
