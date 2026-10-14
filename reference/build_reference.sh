#!/usr/bin/env bash
# Builds Tissue's simulator at the reference commit into reference/build/.
#
# Usage: reference/build_reference.sh [REPO_URL]
#
# Needs git, make, g++ and the Boost headers (Ubuntu: apt-get install libboost-dev).
# The reference data in tests/data/cases/ were generated with g++ 13.3 and the
# Makefile's default flags (-O3 -DNDEBUG). The commit is upstream e74e457, also
# reachable in the supersubscript/tissue fork (before its fix PRs).
set -euo pipefail

commit=e74e457869410e634a474a68f1071ceb14df753e
repo=${1:-https://gitlab.com/slcu/teamHJ/tissue.git}
build=$(cd "$(dirname "$0")" && pwd)/build

if [[ ! -d $build/.git ]]; then
    git clone "$repo" "$build"
fi
git -C "$build" fetch --quiet origin "$commit" || true
git -C "$build" checkout --quiet "$commit"
make -C "$build/src" -j"$(nproc)" ../bin/simulator
echo "Built $build/bin/simulator at $commit"
