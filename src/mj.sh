#!/bin/bash

for x in 1 2 3 4 5 6 7 8 9 10
do
	mkdir -p ../log/mj/
	python main.py reuters cvdd_Net ../log/mj ../data --device cpu --seed $x --clean_txt --embedding_size 300 --pretrained_model GloVe_6B --ad_score context_dist_mean --n_attention_heads 3 --attention_size 150 --lambda_p 1.0 --alpha_scheduler logarithmic --n_epochs 100 --lr 0.01 --lr_milestone 40  --normal_class 2;
	python main.py reuters cvdd_Net ../log/mj ../data --device cpu --seed $x --clean_txt --embedding_size 300 --pretrained_model GloVe_6B --ad_score context_dist_mean --n_attention_heads 3 --attention_size 150 --lambda_p 1.0 --alpha_scheduler soft --n_epochs 100 --lr 0.01 --lr_milestone 40  --normal_class 2;
	python main.py reuters cvdd_Net ../log/mj ../data --device cpu --seed $x --clean_txt --embedding_size 300 --pretrained_model GloVe_6B --ad_score context_dist_mean --n_attention_heads 3 --attention_size 150 --lambda_p 10.0 --alpha_scheduler logarithmic --n_epochs 100 --lr 0.01 --lr_milestone 40  --normal_class 2;
	python main.py reuters cvdd_Net ../log/mj ../data --device cpu --seed $x --clean_txt --embedding_size 300 --pretrained_model GloVe_6B --ad_score context_dist_mean --n_attention_heads 3 --attention_size 150 --lambda_p 10.0 --alpha_scheduler soft --n_epochs 100 --lr 0.01 --lr_milestone 40  --normal_class 2;

done
