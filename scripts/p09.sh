#!/bin/bash

# Default values
VALIDATION_FLAG=""
TEST_FLAG=""
HALF_TEST_FLAG=""

# Parse arguments
for arg in "$@"; do
  case $arg in
    --validation)
      VALIDATION_FLAG="--validation"
      ;;
    --test)
      TEST_FLAG="--test"
      ;;
    --halftest)
      HALF_TEST_FLAG="--halftest"
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: ./p09.sh [--validation] [--test] [--halftest]"
      exit 1
      ;;
  esac
done


export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_THREADING_LAYER=SEQUENTIAL
python scripts/x09__StackNoLeak.py $VALIDATION_FLAG $TEST_FLAG $HALF_TEST_FLAG