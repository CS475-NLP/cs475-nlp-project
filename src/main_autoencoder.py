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

from sklearn.metrics import roc_auc_score

from networks.main import build_network
from autoencoder.model import autoencoder
import torch.optim as optim


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
@click.option('--lr', type=float, default=0.001, help='Initial learning rate for training. Default=0.001')
@click.option('--n_epochs', type=int, default=100, help='Number of epochs to train.')
@click.option('--batch_size', type=int, default=64, help='Batch size for mini-batch training.')
# @click.option('--n_jobs_dataloader', type=int, default=0, help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--n_threads', type=int, default=0,
              help='Sets the number of OpenMP threads used for parallelizing CPU operations')

def main(net_name, dataset_name, data_path, load_config,  tokenizer, clean_txt, normal_class, embedding_size, pretrained_model, attention_size, n_attention_heads, lr, n_epochs, batch_size,seed,device, n_jobs_dataloader, n_threads):
    # Load data
    cfg = Config(locals().copy())

    # Set seed for reproducibility
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        torch.cuda.manual_seed(cfg.settings['seed'])
        torch.backends.cudnn.deterministic = True
        # logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    # logger.info('Computation device: %s' % device)
    # logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)
    if n_threads > 0:
        torch.set_num_threads(n_threads)
        # logger.info('Number of threads used for parallelizing CPU operations: %d' % n_threads)

    ##Data Load##
    dataset = load_dataset(dataset_name, data_path, normal_class, cfg.settings['tokenizer'],
                           clean_txt=cfg.settings['clean_txt'])

    # print('Dataset')
    # print(dataset.train_set.dataset)
    # print(dataset.test_set.dataset)
    # print('Dataset')

    ##Word Embedding##
    embedding = build_network(net_name, dataset, embedding_size=embedding_size, pretrained_model=pretrained_model, update_embedding=False, attention_size=attention_size, n_attention_heads=n_attention_heads)
    # print(embedding)

    AE=autoencoder(embedding)
    train_loader, _ = dataset.loaders(batch_size=cfg.settings['batch_size'], num_workers=0)
    # print(len(train_loader))
    # print(len(test_loader))

    optimizer = optim.Adam(AE.parameters(), lr=cfg.settings['lr'])
    AE.eval()
    for _ in range(cfg.settings['n_epochs']):
        AE.train()
        for data in train_loader:
            optimizer.zero_grad()
            idx, text_batch, label_batch, _ = data
            text_batch, label_batch = text_batch.to('cpu'), label_batch.to('cpu')
            c1, c2, c3, c4, h1, h2, h3, h4 = AE.Encode(text_batch)
            o8 = AE.Decode_train(text_batch, c1, c2, c3, c4, h1, h2, h3, h4)
            loss=AE.Loss(text_batch, o8)
            # print(loss)
            loss.backward()
            optimizer.step()

    _, test_loader = dataset.loaders(batch_size=1, num_workers=0)

    zipped_result = []
    loss_normal = []
    loss_abnormal = []

    # i = 0


    with torch.no_grad():
        for data in test_loader:
            idx, text_batch, label_batch, _ = data
            c1, c2, c3, c4, h1, h2, h3, h4 = AE.Encode(text_batch)
            o8 = AE.Decode_train(text_batch, c1, c2, c3, c4, h1, h2, h3, h4)
            loss = AE.Loss(text_batch, o8)
            # print("zzzz", label_batch)
            # print("zz222", text_batch)
            # print("loooos", loss)
            # if(label_batch[i]==0):
            #     loss_normal.append(loss)
            # elif(label_batch[i]==1):
            #     loss_abnormal.append(loss)
            # print("loss",loss, loss.cpu().data.numpy().tolist())
            # print("label_batch",label_batch, label_batch.cpu().data.numpy().tolist())

            zipped_result += list(zip(idx, label_batch.cpu().data.numpy().tolist(), [loss.cpu().data.numpy().tolist()]))
            # auc_value = roc_auc_score(label_batch, loss)
            # i += 1

    _, labels, scores = zip(*zipped_result)
    labels = np.array(labels)
    scores = np.array(scores)
    # print(labels)
    # print(scores)
    # print(labels.size(), scores.size())

    auc_value = roc_auc_score(labels, scores)
    print('Test AUC: {:.2f}%'.format(100. * auc_value))
    

# main()
if __name__ == '__main__':
    main()
