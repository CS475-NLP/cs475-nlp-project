import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet

class autoencoder(BaseNet):

    def __init__(self, pretrained_model):
        super().__init__()

        # Load pretrained model (which provides a hidden representation per word, e.g. word vector or language model)
        self.pretrained_model = pretrained_model
        self.hidden_size = pretrained_model.embedding_size

    def forward(self, x):
        # x.shape = (sentence_length, batch_size)

        hidden = self.pretrained_model(x)
        print(hidden.shape)
        # hidden.shape = (sentence_length, batch_size, hidden_size)

        return hidden
