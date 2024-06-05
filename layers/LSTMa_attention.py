import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class hidden_attention(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(hidden_attention, self).__init__()


        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.hidden_dim = hidden_size



    def forward(self, x):
        #K，Q的定义
        key = self.key_layer(x)
        query = self.query_layer(x)

        #Q和K.T矩阵相乘
        attention_scores = torch.matmul(query.squeeze(0), key.squeeze(0).T)
        attention_scores = attention_scores / math.sqrt(self.hidden_dim)
        #对每一行进行softmax操作
        attention_probs = F.softmax(attention_scores, dim=-1)


        return attention_probs



