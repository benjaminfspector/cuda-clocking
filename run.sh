#!/bin/bash
nvcc example.cu -arch=sm_86 -o example
./example
python3 analysis.py