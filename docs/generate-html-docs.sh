#!/bin/bash

set -e

# Script to generate html docs
script_dir=$(dirname "$(readlink -f "$0")")

echo "Generating HTML docs from $script_dir"
cd "$script_dir"
rm -rf build
make html
