#!/bin/bash
for x in 0 1 2 3 4 5 
do
       mkdir -p ../log_OCSVM/test_newsgroups20/notfidf/mean/$x/nu0.05
		python main_ocsvm.py newsgroups20 ../log_OCSVM/test_newsgroups20/notfidf/mean/$x/nu0.05 ../data --seed 1 --kernel linear --nu 0.05 --clean_txt --embedding_size 300 --pretrained_word_vectors GloVe_6B --embedding_reduction mean --normalize_embedding --normal_class $x;
       mkdir -p ../log_OCSVM/test_newsgroups20/notfidf/mean/$x/nu0.1
		python main_ocsvm.py newsgroups20 ../log_OCSVM/test_newsgroups20/notfidf/mean/$x/nu0.1 ../data --seed 1 --kernel linear --nu 0.1 --clean_txt --embedding_size 300 --pretrained_word_vectors GloVe_6B --embedding_reduction mean --normalize_embedding --normal_class $x;
       mkdir -p ../log_OCSVM/test_newsgroups20/notfidf/mean/$x/nu0.2
		python main_ocsvm.py newsgroups20 ../log_OCSVM/test_newsgroups20/notfidf/mean/$x/nu0.2 ../data --seed 1 --kernel linear --nu 0.2 --clean_txt --embedding_size 300 --pretrained_word_vectors GloVe_6B --embedding_reduction mean --normalize_embedding --normal_class $x;
       mkdir -p ../log_OCSVM/test_newsgroups20/notfidf/mean/$x/nu0.5
		python main_ocsvm.py newsgroups20 ../log_OCSVM/test_newsgroups20/notfidf/mean/$x/nu0.5 ../data --seed 1 --kernel linear --nu 0.5 --clean_txt --embedding_size 300 --pretrained_word_vectors GloVe_6B --embedding_reduction mean --normalize_embedding --normal_class $x;
done

