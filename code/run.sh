#!/bin/bash
source ~/.bashrc
IT=200
K=50
A=$1
PROC=$2
OUT_PATH=../experiments/sotu_outputs/rho_${A}/k_${K}/
mkdir -p $OUT_PATH
conda activate py27
python enron.py --input sotu_years.npz --output $OUT_PATH/sotu_topics_${K}_${A}_${IT}-$PROC.npz --priv $A --rank $K --niters $IT > $OUT_PATH/sotu_output_${K}_${A}_${IT}-$PROC.txt
