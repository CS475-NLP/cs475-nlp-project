import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet
from networks.self_attention import SelfAttention

class autoencoder_attention(BaseNet):

    def __init__(self, pretrained_model , attention_size=100, n_attention_heads=3):
        super().__init__()

        # Load pretrained model (which provides a hidden representation per word, e.g. word vector or language model)
        self.pretrained_model = pretrained_model
        self.hidden_size = pretrained_model.embedding_size

        # Set self-attention module
        self.attention_size = attention_size
        self.n_attention_heads = n_attention_heads
        self.self_attention = SelfAttention(hidden_size=self.hidden_size, attention_size=attention_size, n_attention_heads=n_attention_heads)

        #Autoencoder
        # self.fc1 = nn.Linear(300, 150)
        # self.fc2 = nn.Linear(150, 50)
        # self.fc3 = nn.Linear(50, 150)
        # self.fc4 = nn.Linear(150, 300)

        self.fc1 = nn.Linear(300, 150)
        # self.fc3 = nn.Linear(150, 50)
        # self.fc5 = nn.Linear(50, 10)
        
        # self.fc6 = nn.Linear(10, 50)
        # self.fc4 = nn.Linear(50, 150)
        self.fc2 = nn.Linear(150, 300)

        # self.relu = nn.ReLU(10)
        # self.dropout = nn.Dropout(0.05)


#         self.ac1 = nn.ReLU()
#         self.ac2 = nn.ReLU()
#         self.ac3 = nn.ReLU()
#         self.ac4 = nn.ReLU()

#         self.dropout = nn.Dropout(0.3)


    def sentence_Embedding(self, x):
        # x.shape = (sentence_length, batch_size)
        hidden = self.pretrained_model(x)  # hidden.shape = (sentence_length, batch_size, hidden_size)
        if hidden.dim()==2:
            hidden=torch.unsqueeze(hidden, 0)

        M, A = self.self_attention(hidden)
        # print(hidden.shape)
        # print(A.shape)
        # print(M.shape)

        return M

    def forward(self, M):

        o1 = self.fc1(M)

#         o1 = self.dropout(o1)
#         # o1 = self.ac1(o1)

#         o2 = self.fc2(o1)
#         o2 = self.dropout(o2)
#         # o2 = self.ac2(o2)

#         o3 = self.fc3(o2)
#         o3 = self.dropout(o3)
#         # o3 = self.ac3(o3)

#         o4 = self.fc4(o3)
#         # o4 = self.ac4(o4)

        # o1=self.dropout(o1)
        # o1 = self.fc3(o1)
        # o1 = self.dropout(o1)
        # o1 = self.fc4(o1)
        # o1 = self.fc5(o1)
        # o1 = self.dropout(o1)
        # o1 = self.relu(o1)
        # o1 = self.fc6(o1)
        # o1 = self.dropout(o1)
        # o1 = self.fc4(o1)
        # o1 = self.dropout(o1)
        
        o4 = self.fc2(o1)
        # o4 = self.relu(o4)
        # o2 = self.dropout(o2)
        # o3 = self.fc3(o2)
        # o3 = self.dropout(o3)
        # o4 = self.fc4(o3)

        # o5 = self.fc5(o4)
        # o6 = self.fc6(o5)

        return o4

    def Loss(self, M, o6):
        loss = nn.MSELoss()
        MSE_Loss = loss(M, o6)

        return MSE_Loss