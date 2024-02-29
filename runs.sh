python src/train.py -m experiment=e_seb trainer.accelerator="gpu" model.lambd=0.01,0.15,0.2,0.25,0.3,0.5,0.8,1 data.batch_size=256 model.loss.kernel.n_kernels=2,4,5,6,7,8
python src/train.py -m experiment=e_seb trainer.accelerator="gpu" model.lambd=0.01,0.15,0.2,0.25,0.3,0.5,0.8,1 data.batch_size=256 model.loss.kernel.n_kernels=2,4,5,6,7,8 data.flip_domain=true
