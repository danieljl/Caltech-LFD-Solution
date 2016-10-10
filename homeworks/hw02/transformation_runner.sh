#!/bin/bash

THIS_DIR=$(dirname $(readlink -f "$0"))

cd "$THIS_DIR/../.."
python -m homeworks.hw02.transformation
