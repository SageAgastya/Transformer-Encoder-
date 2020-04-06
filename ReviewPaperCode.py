import torch.optim as optim
import torch
import tensorflow as tf
import tensorflow_hub as hub
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import copy

g = tf.Graph()
config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
with g.as_default():
    text_input = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
    embed_text = elmo(text_input, signature='default', as_dict=True)['elmo']
    init_op = tf.group([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
g.finalize()
session = tf.compat.v1.Session(graph=g, config=config)
session.run(init_op)


class Embedder():
    def __init__(self):
        super().__init__()

    def __call__(self, X):
        elmo_embed = torch.Tensor([session.run(embed_text, feed_dict={text_input: i}) for i in X])
        return elmo_embed


class Pad():  # pads words as well as sentences
    def __init__(self, bs=4, sent_cnt=2, word_cnt=80, dim=1024):
        self.dim = dim
        self.word_cnt = word_cnt
        self.bs = bs
        self.embed = Embedder()
        self.pe = PositionalEncoder()
        sent_cnt = sent_cnt + 1
        self.zeros = torch.zeros(bs, sent_cnt, int(word_cnt / 2), dim)
        self.dummy = "<PAD> " * int(word_cnt / 2)

    def __call__(self, input):

        for li in input:
            li.append(self.dummy)

        seqlen = [[len(line.split()) for line in para] for para in input]
        self.embed = self.embed(input)
        for i in range(self.bs):
            self.embed[i] = self.embed[i] + self.pe(self.embed[i])
        # print(self.embed.shape)
        for i in range(len(seqlen)):
            m = 0
            for k in seqlen[i]:
                self.zeros[i][m][:k] = self.embed[i][m][:k]
                m += 1

        self.zeros = self.zeros[:, :2]
        # print(self.zeros)
        self.zeros = self.zeros.view(self.bs, self.word_cnt, -1)
        alpha = self.zeros
        # alpha = torch.sum(self.zeros, dim=1)
        return alpha


class PositionalEncoder(nn.Module):  # passed successfully
    def __init__(self, d_model=1024, max_seq_len=80):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads=8, d_model=1024, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


def attention(q, k, v, d_k=1024, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class FeedForward(nn.Module):  # passed succesfully
    def __init__(self, d_model=1024, d_ff=2048, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):  # passed succesfully
    def __init__(self, d_model=1024, eps=1e-6):
        super().__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model=1024, heads=8, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm()
        self.norm_2 = Norm()
        self.attn = MultiHeadAttention()
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, N=5, heads=8, d_model=1024):
        super().__init__()
        self.N = N
        # self.pad = Pad()
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
        self.linear = nn.Linear(d_model, 3)

    def forward(self, x, mask=None):
        # x = self.pad(x)
        for i in range(self.N):
            x = self.layers[i](x)
        return self.linear(self.norm(x))


train_dataset = [[['rishabh is good at gsgujsih', 'rishabh is great'], ['rishabh doesnt like', 'india won match'],['rishabh doesnt like', 'india won match'],['rishabh doesnt like', 'india won match']],
           torch.tensor([[1, 1, 0], [0, 0, 1],[0, 0, 1],[0, 0, 1]])]

p = Pad()
net = Encoder()
# X, y = Preprocess()                  #import from ReviewPaperPreprocess
# a,b,c,d = DataSplit(X, y)            #import from ReviewPaperPreprocess
# train_dataset = (a,c)
rep = p(train_dataset[0])
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
running_loss = 0.0

for epoch in range(5):
    for i, data in enumerate(rep):

        inputs, labels = data, train_dataset[1][i]
        inputs = inputs.unsqueeze(0)
        labels = labels.unsqueeze(0)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 2 == 1:
            plt.scatter(loss.item(), i)
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
plt.show()
print('Finished Training')