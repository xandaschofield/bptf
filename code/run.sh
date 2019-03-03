#!/bin/bash
source ~/.bashrc
IT=200
K=50
A=0.05
conda activate py27
python enron.py --output no_noise/test_file_${K}_${A}_${IT}.npz --priv $A --niters ${IT}
