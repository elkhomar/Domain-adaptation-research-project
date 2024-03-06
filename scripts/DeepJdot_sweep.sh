# With early stopping
python src/train.py -m experiment=e_ada model.loss.reg_d=1,5,0.5,0.1 model.loss.reg_cl=1,5,0.5,0.1 data.batch_size=32,64,128,512,1028
# Without early stopping
python src/train.py -m  experiment=e_ada model.loss.reg_d=1,5,0.5,0.1 model.loss.reg_cl=1,5,0.5,0.1 data.batch_size=32,64,128,512,1028 trainer.min_epochs=120

