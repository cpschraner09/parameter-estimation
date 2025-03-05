#!/bin/bash

#changed bash file
cd "$(dirname "$0")/.."

python3 -m unittest discover -s tests
