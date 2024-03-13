python src/train.py -m experiment=e_seb trainer.accelerator="gpu" model.lambd=0.7 data.batch_size=512 data.flip_domain=false seed=42,55,35,18,19 model.loss.kernel.n_kernels=8
python src/train.py -m experiment=e_seb trainer.accelerator="gpu" model.lambd=0.1 data.batch_size=256 data.flip_domain=true seed=42,55,35,18,19 model.loss.kernel.n_kernels=3
