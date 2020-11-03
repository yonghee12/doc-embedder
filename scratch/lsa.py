from gensim import corpora, models
import logging

from doc_embedder.modules import *
from doc_embedder.functions import *

from pprint import pprint
from konlpy.tag import Mecab
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


tok = Mecab()

corpus = load_watcha_sample_corpus(n_sample=None)
texts = list(corpus['text'])
len(texts)
allowed_pos = {
    'NNG',
    'NNP',
    # # 'VV',
    # 'VA',
}

tokens_matrix = [[token[0] for token in tok.pos(line) if token[1] in allowed_pos] for line in texts]
# for i in range(len(texts)):
#     print(texts[i])
#     print(tokens_matrix[i])

# unique_tokens = get_uniques_from_nested_lists(tokens_matrix)
# token2idx, idx2token = get_item2idx(unique_tokens)


tokens_matrix_joined = [' '.join(tokens) for tokens in tokens_matrix]
tfidfvectorizer = TfidfVectorizer(max_features=1000, max_df=0.5, smooth_idf=True)
countvectorizer = CountVectorizer()
X_count = countvectorizer.fit_transform(tokens_matrix_joined)
X_tfidf = tfidfvectorizer.fit_transform(tokens_matrix_joined)

svd_model = TruncatedSVD(n_components=30, algorithm='randomized', n_iter=100, random_state=122)
svd_model.fit(X_count)
# svd_model.fit(X_tfidf)
len(svd_model.components_)

terms = countvectorizer.get_feature_names()


def get_topics(components, feature_names, n=10):
    for idx, topic in enumerate(components):
        print("Topic %d:" % (idx + 1), [(feature_names[i], topic[i].round(2)) for i in topic.argsort()[:-n - 1:-1]])

get_topics(svd_model.components_, terms)
# corpus = corpora.MmCorpus('/tmp/deerwester.mm')  # load a corpus of nine documents, from the Tutorials
# id2word = corpora.Dictionary.load('/tmp/deerwester.dict')

# run distributed LSA on nine documents
