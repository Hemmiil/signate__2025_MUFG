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
      echo "Usage: ./p04.sh [--validation] [--test]"
      exit 1
      ;;
  esac
done

python scripts/x04__ML.py $VALIDATION_FLAG $TEST_FLAG
