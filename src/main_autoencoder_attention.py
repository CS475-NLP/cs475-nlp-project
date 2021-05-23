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
from autoencoder.model_attention import autoencoder_attention
import torch.optim as optim
from sklearn.metrics import roc_auc_score

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
@click.option('--n_attention_heads', type=int, default=3, help='Number of attention heads in self-attention module.')

def main(net_name, dataset_name, data_path, load_config,  tokenizer, clean_txt, normal_class, embedding_size, pretrained_model, attention_size, n_attention_heads):
    # Load data
    cfg = Config(locals().copy())

    ##Data Load##
    dataset = load_dataset(dataset_name, data_path, normal_class, cfg.settings['tokenizer'],
                           clean_txt=cfg.settings['clean_txt'])

    # print('Dataset')
    # print(dataset.train_set.dataset)
    # print(dataset.test_set.dataset)
    # print('Dataset')

    ##Word Embedding##
    embedding = build_network(net_name, dataset, embedding_size=embedding_size, pretrained_model=pretrained_model, update_embedding=False, attention_size=attention_size, n_attention_heads=n_attention_heads)
    print(embedding)

    AE=autoencoder_attention(embedding)
    train_loader, test_loader = dataset.loaders(batch_size=32, num_workers=0)
    # print(len(train_loader))
    # print(len(test_loader))



    optimizer = optim.Adam(AE.parameters(), lr=0.02)
    for i in range(100):
        AE.train()
        for data in train_loader:
            optimizer.zero_grad()
            idx, text_batch, label_batch, _ = data
            text_batch, label_batch = text_batch.to('cpu'), label_batch.to('cpu')

            M = AE.sentence_Embedding(text_batch)
            M_recon = AE.forward(M)
            loss=AE.Loss(M, M_recon)
            print(loss)

            loss.backward()
            optimizer.step()

    train_loader, test_loader = dataset.loaders(batch_size=1, num_workers=0)

    zipped_result = []
    loss_normal = []
    loss_abnormal = []

    for data in test_loader:
        idx, text_batch, label_batch, _ = data
        M = AE.sentence_Embedding(text_batch)
        M_recon = AE.forward(M)
        loss = AE.Loss(M, M_recon)

        print(loss)
        zipped_result += list(zip(idx, label_batch.cpu().data.numpy().tolist(), [loss.cpu().data.numpy().tolist()]))
        print(label_batch)
    _, labels, scores = zip(*zipped_result)
    labels = np.array(labels)
    scores = np.array(scores)

    auc_value = roc_auc_score(labels, scores)
    print('Test AUC: {:.2f}%'.format(100. * auc_value))

if __name__ == '__main__':
    main()
