# Domain-adaptation-research-project
Implementation of several domain adaptation methods with a pipeline that allows to run complex experiments with a CLI using yaml config files (hydra library)

## How It Works :
The base configuration is found in config/e_ada.yaml and can be overridden by passing arguments in the command line :
- For regular training : python src/train.py  experiment=e_ada model.lambd=1.0 model.loss._target_=src.losses.SlicedWassersteinLoss
- For grid search : python src/train.py -m  experiment=e_ada model.lambd=0.5,1.0,1.5,2.0 data.batch_size=16,32,64
- For profiling: python src/train.py  experiment=e_ada model.lambd=1.0 model.loss._target_=src.losses.SlicedWassersteinLoss debug=profiler

The pipeline is based on https://github.com/ashleve/lightning-hydra-template

## Overleaf :
https://www.overleaf.com/project/65bbbd12e2b5ee9e67df749c
## Wandb runs :
- Omar : https://wandb.ai/omarel/Domain%20adaptation%20research%20project
- Seb : https://wandb.ai/sebvol/Domain%20adaptation%20research%20project?workspace=user-sebastien-vol

# Methods :
## Invariant :
- Deep Coral
- Wasserstein (Sliced, Unbiased minibatch wass, Sinkhorn, Wgan loss)
- MMD (https://arxiv.org/pdf/1412.3474.pdf)
- Deep Jdot
- TTT++ (https://proceedings.neurips.cc/paper_files/paper/2021/file/b618c3210e934362ac261db280128c22-Paper.pdf)


# Data :
- Digit-five (MNIST-USPS): https://drive.usercontent.google.com/download?id=1A4RJOFj4BJkmliiEL7g9WzNIDUHLxfmm&export=download&authuser=0
- MNIST-M https://drive.google.com/file/d/0B_tExHiYS-0veklUZHFYT19KYjg/view?pli=1&resourcekey=0-DE7ivVVSn8PwP9fdHvSUfA
- Toy datasets with different shifts from Skada

# Useful resources

https://remi.flamary.com/cours/tuto_da/DA_shallow_to_deep.pdf
