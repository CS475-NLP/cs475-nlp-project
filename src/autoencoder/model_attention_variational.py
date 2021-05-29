import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet
from networks.self_attention import SelfAttention

import numpy as np

class vae(BaseNet):

    def __init__(self, pretrained_model , attention_size=100, n_attention_heads=3):
        super().__init__()

        # Load pretrained model (which provides a hidden representation per word, e.g. word vector or language model)
        self.pretrained_model = pretrained_model
        self.hidden_size = pretrained_model.embedding_size

        # Set self-attention module
        self.attention_size = attention_size
        self.n_attention_heads = n_attention_heads
        self.self_attention = SelfAttention(hidden_size=self.hidden_size, attention_size=attention_size, n_attention_heads=n_attention_heads)
        self.sig = nn.Sigmoid()

        #Encoder
        # self.enc1 = nn.Linear(300, 150)
        # self.en_ac1 = nn.ReLU()
        # self.enc2 = nn.Linear(150, 50)
        # self.en_ac2 = nn.ReLU()

        self.mean = nn.Linear(300, 150)
        self.mean_ac = nn.ReLU()
        self.log_var = nn.Linear(300, 150)
        self.log_var_ac = nn.ReLU()

        # Decoder
        # self.dec1 = nn.Linear(10, 50)
        # self.de_ac1 = nn.ReLU()
        # self.dec2 = nn.Linear(50, 150)
        # self.de_ac2 = nn.ReLU()
        self.dec3 = nn.Linear(150, 300)
        self.de_ac3 = nn.Sigmoid()

        ##Dropout
        self.dropout = nn.Dropout(0)

    def sentence_Embedding(self, x):
        # x.shape = (sentence_length, batch_size)
        hidden = self.pretrained_model(x)  # hidden.shape = (sentence_length, batch_size, hidden_size)
        if hidden.dim()==2:
            hidden=torch.unsqueeze(hidden, 0)

        M, A = self.self_attention(hidden)
        M=self.sig(M)
        # print(hidden.shape)
        # print(A.shape)
        # print(M.shape)

        return M

    def Encode(self, M):

        # o1 = self.enc1(M)
        # o1 = self.dropout(o1)
        # o1 = self.en_ac1(o1)
        #
        # o2 = self.enc2(o1)
        # o2 = self.dropout(o2)
        # o2 = self.en_ac2(o2)

        mean = self.mean(M)
        mean = self.mean_ac(mean)

        log_var = self.log_var(M)
        log_var = self.log_var_ac(log_var)

        return mean, log_var

    def reparametrize(self, mean, log_var, batch):

        # head=3
        # feature=10
        # eps= np.random.normal(0, 1, (batch, head, feature))
        # z = mean + np.exp(log_var/2) * eps

        z = torch.normal(mean, torch.exp(log_var/2))

        return z

    def decode(self, z):

        # o1 = self.dec1(z)
        # o1 = self.dropout(o1)
        # o1 = self.de_ac1(o1)
        #
        # o2 = self.dec2(o1)
        # o2 = self.dropout(o2)
        # o2 = self.de_ac2(o2)

        o3 = self.dec3(z)
        o3 = self.dropout(o3)
        M_recon = self.de_ac3(o3)

        return M_recon


    def Recon_Loss(self, M, M_recon):
        # loss = nn.functional.binary_cross_entropy(M, M_recon)
        term1= - torch.mul(M, torch.log(M_recon))
        term2= - torch.mul(1-M, torch.log(1-M_recon))

        bce_loss= term1+term2
        loss= torch.mean(bce_loss)

        return loss

    def KL_divergence(self, mean, log_var):
        KL_divergence = -0.5 * (1 + log_var - torch.mul(mean, mean) - torch.exp(log_var))
        KL_divergence = torch.mean(KL_divergence)

        return KL_divergence