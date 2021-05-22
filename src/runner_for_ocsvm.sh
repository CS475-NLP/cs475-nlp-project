#!/bin/bash

# for reuters
for x in 0 1 2 3 4 5 6
do
	mkdir -p ../log/test_reuters/ocsvm/mean/nu_1
        python main_ocsvm.py reuters ../log/test_reuters/ocsvm/mean/nu_1 ../data  --seed 1 --clean_txt --embedding_size 300 --pretrained_word_vectors GloVe_6B --nu 0.05 --embedding_reduction mean --normalize_embedding True --use_tfidf_weights False --normal_class $x;
	mkdir -p ../log/test_reuters/ocsvm/mean/nu_2
        python main_ocsvm.py reuters ../log/test_reuters/ocsvm/mean/nu_2 ../data  --seed 1 --clean_txt --embedding_size 300 --pretrained_word_vectors GloVe_6B --nu 0.1 --embedding_reduction mean --normalize_embedding True --use_tfidf_weights False --normal_class $x;
	mkdir -p ../log/test_reuters/ocsvm/mean/nu_3
        python main_ocsvm.py reuters ../log/test_reuters/ocsvm/mean/nu_3 ../data  --seed 1 --clean_txt --embedding_size 300 --pretrained_word_vectors GloVe_6B --nu 0.2 --embedding_reduction mean --normalize_embedding True --use_tfidf_weights False --normal_class $x;
	mkdir -p ../log/test_reuters/ocsvm/mean/nu_4
        python main_ocsvm.py reuters ../log/test_reuters/ocsvm/mean/nu_4 ../data  --seed 1 --clean_txt --embedding_size 300 --pretrained_word_vectors GloVe_6B --nu 0.5 --embedding_reduction mean --normalize_embedding True --use_tfidf_weights False --normal_class $x;
done

for x in 0 1 2 3 4 5 6
do
        mkdir -p ../log/test_reuters/ocsvm/max/nu_1
        python main_ocsvm.py reuters ../log/test_reuters/ocsvm/max/nu_1 ../data  --seed 1 --clean_txt --embedding_size 300 --pretrained_word_vectors GloVe_6B --nu 0.05 --embedding_reduction max --normalize_embedding True --use_tfidf_weights False --normal_class $x;
        mkdir -p ../log/test_reuters/ocsvm/max/nu_2
        python main_ocsvm.py reuters ../log/test_reuters/ocsvm/max/nu_2 ../data  --seed 1 --clean_txt --embedding_size 300 --pretrained_word_vectors GloVe_6B --nu 0.1 --embedding_reduction max --normalize_embedding True --use_tfidf_weights False --normal_class $x;
        mkdir -p ../log/test_reuters/ocsvm/max/nu_3
        python main_ocsvm.py reuters ../log/test_reuters/ocsvm/max/nu_3 ../data  --seed 1 --clean_txt --embedding_size 300 --pretrained_word_vectors GloVe_6B --nu 0.2 --embedding_reduction max --normalize_embedding True --use_tfidf_weights False --normal_class $x;
        mkdir -p ../log/test_reuters/ocsvm/max/nu_4
        python main_ocsvm.py reuters ../log/test_reuters/ocsvm/max/nu_4 ../data  --seed 1 --clean_txt --embedding_size 300 --pretrained_word_vectors GloVe_6B --nu 0.5 --embedding_reduction max --normalize_embedding True --use_tfidf_weights False --normal_class $x;
done

## for news
for x in 0 1 2 3 4 5
do
        mkdir -p ../log/test_newsgroup20/ocsvm/mean/nu_1
        python main_ocsvm.py newsgroups20 ../log/test_newsgroups20/ocsvm/mean/nu_1 ../data  --seed 1 --clean_txt --embedding_size 300 --pretrained_word_vectors GloVe_6B --nu 0.05 --embedding_reduction mean --normalize_embedding True --use_tfidf_weights False --normal_class $x;
        mkdir -p ../log/test_newsgroup20/ocsvm/mean/nu_2
        python main_ocsvm.py newsgroups20 ../log/test_newsgroups20/ocsvm/mean/nu_2 ../data  --seed 1 --clean_txt --embedding_size 300 --pretrained_word_vectors GloVe_6B --nu 0.1 --embedding_reduction mean --normalize_embedding True --use_tfidf_weights False --normal_class $x;
        mkdir -p ../log/test_newsgroup20/ocsvm/mean/nu_3
        python main_ocsvm.py newsgroups20 ../log/test_newsgroups20/ocsvm/mean/nu_3 ../data  --seed 1 --clean_txt --embedding_size 300 --pretrained_word_vectors GloVe_6B --nu 0.2 --embedding_reduction mean --normalize_embedding True --use_tfidf_weights False --normal_class $x;
        mkdir -p ../log/test_newsgroup20/ocsvm/mean/nu_4
        python main_ocsvm.py newsgroups20 ../log/test_newsgroups20/ocsvm/mean/nu_4 ../data  --seed 1 --clean_txt --embedding_size 300 --pretrained_word_vectors GloVe_6B --nu 0.5 --embedding_reduction mean --normalize_embedding True --use_tfidf_weights False --normal_class $x;
done

for x in 0 1 2 3 4 5
do
        mkdir -p ../log/test_newsgroups20/ocsvm/max/nu_1
        python main_ocsvm.py newsgroups20 ../log/test_newsgroups20/ocsvm/max/nu_1 ../data  --seed 1 --clean_txt --embedding_size 300 --pretrained_word_vectors GloVe_6B --nu 0.05 --embedding_reduction max --normalize_embedding True --use_tfidf_weights False --normal_class $x;
        mkdir -p ../log/test_newsgroups20/ocsvm/max/nu_2
        python main_ocsvm.py newsgroups20 ../log/test_newsgroups20/ocsvm/max/nu_2 ../data  --seed 1 --clean_txt --embedding_size 300 --pretrained_word_vectors GloVe_6B --nu 0.1 --embedding_reduction max --normalize_embedding True --use_tfidf_weights False --normal_class $x;
        mkdir -p ../log/test_newsgroups20/ocsvm/max/nu_3
        python main_ocsvm.py newsgroups20 ../log/test_newsgroups20/ocsvm/max/nu_3 ../data  --seed 1 --clean_txt --embedding_size 300 --pretrained_word_vectors GloVe_6B --nu 0.2 --embedding_reduction max --normalize_embedding True --use_tfidf_weights False --normal_class $x;
        mkdir -p ../log/test_newsgroups20/ocsvm/max/nu_4
        python main_ocsvm.py newsgroups20 ../log/test_newsgroups20/ocsvm/max/nu_4 ../data  --seed 1 --clean_txt --embedding_size 300 --pretrained_word_vectors GloVe_6B --nu 0.5 --embedding_reduction max --normalize_embedding True --use_tfidf_weights False --normal_class $x;
done
