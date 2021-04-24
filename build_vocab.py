from collections import Counter
from torchtext.vocab import GloVe
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def build_dictionary(texts, vocab_size):
    counter = Counter()
    SPECIAL_TOKENS = ['<PAD>', '<UNK>']

    for word in texts:
        counter.update(word)

    words = [word for word, count in counter.most_common(vocab_size - len(SPECIAL_TOKENS))]
    embeddings = None
    embed_pretrained = Glove(name='6B')
    for word in words:
        if embeddings:
            embeddings = torch.cat((embeddings,embed_pretrained[word]),dim=1)
    words = SPECIAL_TOKENS + words
    word2idx = {word: idx for idx, word in enumerate(words)}


    # sem_embedding = None
    # syn_embedding = None
    # with tf.Session() as sess:
    #     saver = tf.train.Saver()
    #     saver.restore('./tools/syntactic_semantic_embeddings/embeddings/syntactic_embeddings')
    #     syn_embedding = sess.run()
    #     syn_embedding = tf.convert_to_tensor(syn_embedding)
    #     saver.restore('./tools/syntactic_semantic_embeddings/embeddings/semantic_embeddings')
    #     sem_embedding = sess.run()
    #     sem_embedding = tf.convert_to_tensor(sem_embedding)
    # print('syn embedding size',syn_embedding.size())
    # print('sem embedding size',sem_embedding.size())

    return word2idx, embeddings