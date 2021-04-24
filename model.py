import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import GloVe
import tensorflow as tf

class RCNN(nn.Module):
    """
    Recurrent Convolutional Neural Networks for Text Classification (2015)
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, hidden_size_linear, class_num, dropout,
                 use_lexical=True,use_syntactic=True,use_semantic=True):
        super(RCNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(GloVe(name='840B', dim=embedding_dim)) if use_lexical else nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.sem_embedding = None
        self.syn_embedding = None
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore('./tools/syntactic_semantic_embeddings/embeddings/syntactic_embeddings')
            self.syn_embedding = sess.run()
            self.syn_embedding = tf.convert_to_tensor(self.syn_embedding)
            saver.restore('./tools/syntactic_semantic_embeddings/embeddings/semantic_embeddings')
            self.sem_embedding = sess.run()
            self.sem_embedding = tf.convert_to_tensor(self.sem_embedding)
        print('syn embedding size',self.syn_embedding.size())
        print('sem embedding size',self.sem_embedding.size())
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True, dropout=dropout)
        self.W = nn.Linear(embedding_dim + 2*hidden_size, hidden_size_linear)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(hidden_size_linear, class_num)

    def forward(self, x):
        # x = |bs, seq_len|
        x_emb = self.embedding(x)
        # x_emb = |bs, seq_len, embedding_dim|
        output, _ = self.lstm(x_emb)
        # output = |bs, seq_len, 2*hidden_size|
        output = torch.cat([output, x_emb], 2)
        # output = |bs, seq_len, embedding_dim + 2*hidden_size|
        output = self.tanh(self.W(output)).transpose(1, 2)
        # output = |bs, seq_len, hidden_size_linear| -> |bs, hidden_size_linear, seq_len|
        output = F.max_pool1d(output, output.size(2)).squeeze(2)
        # output = |bs, hidden_size_linear|
        output = self.fc(output)
        # output = |bs, class_num|
        return output