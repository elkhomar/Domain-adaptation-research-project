#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

#python src/train.py -m  experiment=e_ada model.lambd=0.5,1.0,1.5,2.0 data.batch_size=16,32,64
python train.py -m hparams_search=lambda_search experiment=e_ada data.flip_domain=True logger.wandb.group="unbiased Wass usps to mnist"
python train.py -m hparams_search=lambda_search experiment=e_ada data.flip_domain=False logger.wandb.group="unbiased Wass mnist to usps"