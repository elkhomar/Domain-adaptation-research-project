#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py -m  experiment=e_ada model.lambd=0.5,1.0,1.5,2.0 data.batch_size=16,32,64