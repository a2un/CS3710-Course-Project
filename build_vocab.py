from collections import Counter
import torch
from torchtext.vocab import GloVe
import numpy as np
import re


def build_dictionary(texts, vocab_size, lexical, syntactic, semantic):
    sem_embed_path = './tools/syntactic_semantic_embeddings/WordGCN/embeddings/semantic_embedding'
    syn_embed_path = './tools/syntactic_semantic_embeddings/WordGCN/embeddings/syntactic_embedding'
    lex_embed = GloVe(name='6B')
    counter = Counter()
    SPECIAL_TOKENS = ['<PAD>', '<UNK>']

    for word in texts:
        counter.update(word)
    
    words = [word for word, count in counter.most_common(vocab_size - len(SPECIAL_TOKENS))]
    
    embedding = None
    for word in words:
        if not(embedding == None):
            embedding = torch.cat((embeddings,lex_embed[word]),dim=1)
        else:
            embedding = lex_embed[word]

    words = SPECIAL_TOKENS + words
    word2idx = {word: idx for idx, word in enumerate(words)}

    sem_embedding = torch.from_numpy(np.array([[float(eval(score,{},{})) if bool(re.match(r'^-?\d+(\.\d+)?$', score)) else 0 for score in line.split()[1:]] for line in open(sem_embed_path,'r').readlines()[:vocab_size]]))
    syn_embedding = torch.from_numpy(np.array([[float(eval(score,{},{})) if bool(re.match(r'^-?\d+(\.\d+)?$', score)) else 0 for score in line.split()[1:]] for line in open(syn_embed_path,'r').readlines()[:vocab_size]]))

    print('lex embedding size',embedding.size())
    print('syn embedding size',syn_embedding.size())
    print('sem embedding size',sem_embedding.size())

    tpl = (embedding + syn_embedding + sem_embedding) if lexical and semantic and syntactic \
                            else syntactic_embedding + semantic_embedding if semantic and syntactic \
                            else (embedding + semantic_embedding) if lexical and semantic \
                            else (embedding + syntactic_embedding) if lexical and syntactic \
                            else embedding if lexical \
                            else syn_embedding if syntactic \
                            else sem_embedding if semantic \
                            else None


    return word2idx, tpl