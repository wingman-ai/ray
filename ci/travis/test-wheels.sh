#!/usr/bin/env bash

# Cause the script to exit if a single command fails.
set -e

# Show explicitly which commands are currently running.
set -x

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)

platform="unknown"
unamestr="$(uname)"
if [[ "$unamestr" == "Linux" ]]; then
  echo "Platform is linux."
  platform="linux"
else
  echo "Unrecognized platform."
  exit 1
fi

TEST_SCRIPT=${ROOT_DIR}/../../python/ray/tests/test_microbenchmarks.py

if [[ "$platform" == "linux" ]]; then
  # Now test Python 3.6.

  # Install miniconda.
  wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O miniconda3.sh
  bash miniconda3.sh -b -p $HOME/miniconda3

  # Find the right wheel by grepping for the Python version.
  PYTHON_WHEEL=$(find ${ROOT_DIR}/../../.whl -type f -maxdepth 1 -print | grep -m1 '36')

  # Install the wheel.
  $HOME/miniconda3/bin/pip install ${PYTHON_WHEEL}

  # Run a simple test script to make sure that the wheel works.
  $HOME/miniconda3/bin/python ${TEST_SCRIPT}
fi
