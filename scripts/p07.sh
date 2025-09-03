#!/bin/bash

# Default values
VALIDATION_FLAG=""
TEST_FLAG=""

# Parse arguments
for arg in "$@"; do
  case $arg in
    --validation)
      VALIDATION_FLAG="--validation"
      ;;
    --test)
      TEST_FLAG="--test"
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: ./p07.sh [--validation] [--test]"
      exit 1
      ;;
  esac
done


export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_THREADING_LAYER=SEQUENTIAL
python scripts/x07__Stack.py $VALIDATION_FLAG $TEST_FLAG