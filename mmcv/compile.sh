#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

echo "Building sync batchnorm op..."
cd sync_batchnorm/
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace