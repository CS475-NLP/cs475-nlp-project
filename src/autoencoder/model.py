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

        self.lstm1 = nn.LSTM(input_size=300, hidden_size=150)
        self.lstm2 = nn.LSTM(input_size=150, hidden_size=50)
        self.lstm3 = nn.LSTM(input_size=50, hidden_size=150)
        self.lstm4 = nn.LSTM(input_size=150, hidden_size=300)

        self.lstm5 = nn.LSTM(input_size=300, hidden_size=150)
        self.lstm6 = nn.LSTM(input_size=150, hidden_size=50)
        self.lstm7 = nn.LSTM(input_size=50, hidden_size=150)
        self.lstm8 = nn.LSTM(input_size=150, hidden_size=300)

    def forward(self, x):
        # x.shape = (sentence_length, batch_size)

        hidden = self.pretrained_model(x)
        print(hidden.shape)
        # hidden.shape = (sentence_length, batch_size, hidden_size)

        return hidden
    def Encode(self,x):
        hidden = self.pretrained_model(x)
        o1, (c1,h1) = self.lstm1(hidden)
        o2, (c2, h2) = self.lstm2(o1)
        o3, (c3, h3) = self.lstm3(o2)
        o4, (c4, h4) = self.lstm4(o3)

        return c1, c2, c3, c4, h1, h2, h3, h4

    def Decode_Train(self, x, c1, c2, c3, c4, h1, h2, h3, h4):
        hidden = self.pretrained_model(x)
        o5, (c5, h5) = self.lstm5(hidden, (c1, h1))
        o6, (c6, h6) = self.lstm6(o5, (c2, h2))
        o7, (c7, h7) = self.lstm7(o6, (c3, h3))
        o8, (c8, h8) = self.lstm8(o7, (c4, h4))

        return o8

    def Loss(self, x, o8):
        hidden = self.pretrained_model(x)
        loss = nn.MSELoss()
        MSE_Loss = loss(o8, hidden)

        return MSE_Loss



    def Decode_Test(self, y):
        print('b')
