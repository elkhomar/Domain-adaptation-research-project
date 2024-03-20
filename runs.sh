python src/train.py -m experiment=e_seb trainer.accelerator="gpu" model.lambd=0.7 data.batch_size=512 data.flip_domain=false model.loss.kernel.n_kernels=2,4,6,8,10,12 

