#!/bin/bash

# Check if npy file path is provided
if [ -z "$1" ]; then
  echo "Usage: ./p03.sh <path_to_npy_file>"
  exit 1
fi

NPY_PATH=$1

# Prompt for memo text interactively
echo "input memo!"
read MEMO_TEXT

python scripts/x03__CrtSubmission.py "$NPY_PATH" "$MEMO_TEXT"