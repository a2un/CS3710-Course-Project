from collections import Counter
from torchtext.vocab import GloVe
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def build_dictionary(texts, vocab_size, lexical, syntactic, semantic):
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


    sem_embedding = tf.Variable(tf.random_normal(shape=[300]),'sem_embedding')
    sem_embed_path = './tools/syntactic_semantic_embeddings/WordGCN/embeddings/semantic_embedding'
    # syn_embedding = None
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(sem_embed_path)#tf.train.Saver([sem_embedding])
    #     saver.restore(sess,'./tools/syntactic_semantic_embeddings/embeddings/syntactic_embeddings')
    #     syn_embedding = sess.run()
    #     syn_embedding = tf.convert_to_tensor(syn_embedding)
        saver.restore(sess,tf.train.latest_checkpoint(sem_embed_path))
        sem_embedding = sess.run('sem_embedding:0')
        sem_embedding = tf.convert_to_tensor(sem_embedding)
    # print('syn embedding size',syn_embedding.size())
    print('sem embedding size',sem_embedding.size())

    return word2idx, GloVe(name='6B')