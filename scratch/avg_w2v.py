import pickle
import pandas as pd
import numpy as np
import gensim.models
from string import punctuation
from direct_redis import DirectRedis

from _config import *
from doc_embedder.modules import *
from doc_embedder.functions import *

path = os.path.join('/home/ubuntu/yonghee/doc-embedder/data', 'corpora_mecab_938.pk')
# corpora = pickle.load(path)
corpora = pd.read_pickle(path)
print(len(corpora))

r = DirectRedis(**BOOK_ID_MAPPING)


def get_book_name_from_book_id(book_id):
    return r.hget("book_id", f'{book_id}:book_name')


tok = Mecab()

corpus = load_watcha_sample_corpus(n_sample=None)
texts = list(corpus['text'])
print(len(texts))
allowed_pos = {
    'NNG',
    'NNP',
    # # 'VV',
    # 'VA',
}
allowed_pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣]+')
texts = [re.sub(r"([.?!])\1+", r'\1', line) for line in texts]
texts = [allowed_pattern.sub(' ', line).strip() for line in texts]
tokens_matrix = [[token[0] for token in tok.pos(line) if token[1] in allowed_pos] for line in texts]
model = gensim.models.Word2Vec(sentences=tokens_matrix)

del texts
del tokens_matrix

corpus_agg = corpus.groupby(by='title').sum()


def get_vector(model, word):
    if word in model.wv.vocab:
        return model.wv.vectors[model.wv.vocab[word].index]
    else:
        return None


def get_vectors_from_tokens(model, tokens, pooling=None):
    vectors = []
    for token in tokens:
        vector = get_vector(model, token)
        if vector is None:
            continue
        vectors.append(vector)

    if pooling == 'average' or pooling == 'avg':
        if not vectors:
            return None
        return np.array(vectors).mean(axis=0)

    return vectors


allowed_pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣]+')

texts = corpus_agg['text']
texts = [re.sub(r"([.?!])\1+", r'\1', line) for line in texts]
texts = [allowed_pattern.sub(' ', line).strip() for line in texts]
tokens_matrix = [[token[0] for token in tok.pos(line) if token[1] in allowed_pos] for line in texts]

wv_mat = []
for tokens in tokens_matrix:
    vector = get_vectors_from_tokens(model, tokens, pooling='average')
    if vector is None:
        raise Exception("no vector")
        # continue
    wv_mat.append(vector)

wv_mat = np.array(wv_mat)
print(wv_mat.shape)

# wv_mat = np.random.random((5, 20))
norms = np.linalg.norm(wv_mat, axis=1, keepdims=True)
cos_sim = (wv_mat @ wv_mat.T) / (norms @ norms.T)
# cos_sim
df = pd.DataFrame(cos_sim, columns=corpus_agg.index, index=corpus_agg.index)
corpus_agg.index
print(df)
filename = "movie_agg.csv"
path = os.path.join(ROOT, filename)
df.to_csv(path, encoding='utf8')
print('hellaosdf')
