import click
import torch
import logging
import random
import numpy as np

from utils.config import Config
from utils.visualization import plot_matrix_heatmap, plot_joyplot
from utils.misc import print_text_samples, print_top_words, get_correlation_matrix
from cvdd import CVDD
from datasets.main import load_dataset

from networks.main import build_network
from autoencoder.model import autoencoder


################################################################################
# Settings
################################################################################
@click.command()
@click.argument('net_name', type=click.Choice(['cvdd_Net','autoencoder']))
@click.argument('dataset_name', type=click.Choice(['reuters', 'newsgroups20', 'imdb']))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--tokenizer', default='spacy', type=click.Choice(['spacy', 'bert']), help='Select text tokenizer.')
@click.option('--clean_txt', is_flag=True, help='Specify if text should be cleaned in a pre-processing step.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
@click.option('--embedding_size', type=int, default=None, help='Size of the word vector embedding.')
@click.option('--pretrained_model', default=None,
              type=click.Choice([None, 'GloVe_6B', 'GloVe_42B', 'GloVe_840B', 'GloVe_twitter.27B', 'FastText_en',
                                 'bert']),
              help='Load pre-trained word vectors or language models to initialize the word embeddings.')
@click.option('--attention_size', type=int, default=100, help='Self-attention module dimensionality.')
@click.option('--n_attention_heads', type=int, default=1, help='Number of attention heads in self-attention module.')

def main(net_name, dataset_name, data_path, load_config,  tokenizer, clean_txt, normal_class, embedding_size, pretrained_model, attention_size, n_attention_heads):
    # Load data
    cfg = Config(locals().copy())

    ##Data Load##
    dataset = load_dataset(dataset_name, data_path, normal_class, cfg.settings['tokenizer'],
                           clean_txt=cfg.settings['clean_txt'])

    print('Dataset')
    print(dataset.train_set.dataset)
    print(dataset.test_set.dataset)
    print('Dataset')

    ##Word Embedding##
    embedding= build_network(net_name, dataset, embedding_size=embedding_size, pretrained_model=pretrained_model, update_embedding=False, attention_size=attention_size, n_attention_heads=n_attention_heads)
    print(embedding)

    AE=autoencoder(embedding)

    train_loader, _ = dataset.loaders(batch_size=64, num_workers=0)

    for data in train_loader:
        idx, text_batch, label_batch, _ = data
        text_batch, label_batch = text_batch.to('cpu'), label_batch.to('cpu')
        c1, c2, c3, c4, h1, h2, h3, h4 = AE.Encode(text_batch)
        o8 = AE.Decode_Train(text_batch, c1, c2, c3, c4, h1, h2, h3, h4)


if __name__ == '__main__':
    main()
