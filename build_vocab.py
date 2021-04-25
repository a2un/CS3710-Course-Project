from collections import Counter
import torch
from torchtext.vocab import GloVe
import numpy as np


def build_dictionary(texts, vocab_size, lexical, syntactic, semantic):
    sem_embed_path = './tools/syntactic_semantic_embeddings/WordGCN/embeddings/semantic_embedding'
    syn_embed_path = './tools/syntactic_semantic_embeddings/WordGCN/embeddings/syntactic_embedding'
    
    counter = Counter()
    SPECIAL_TOKENS = ['<PAD>', '<UNK>']

    for word in texts:
        counter.update(word)

    words = [word for word, count in counter.most_common(vocab_size - len(SPECIAL_TOKENS) if not(lexical) else 0)]
    # for word in words:
    #     if embeddings:
    #         embeddings = torch.cat((embeddings,embed_pretrained[word]),dim=1)
    words = words if lexical else SPECIAL_TOKENS + words
    word2idx = {word: idx for idx, word in enumerate(words)}

    sem_embed = torch.from_numpy(np.array([[float(score) for score in line.split()[1:]] for line in open(sem_embed_path,'r').readlines()]))
    syn_embed = torch.from_numpy(np.array([[float(score) for score in line.split()[1:]] for line in open(sem_embed_path,'r').readlines()]))

    print('syn embedding size',syn_embedding.size())
    print('sem embedding size',sem_embedding.size())

    return word2idx, GloVe(name='6B'), syn_embed, sem_embed