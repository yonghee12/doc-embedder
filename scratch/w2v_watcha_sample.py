import sys
from pathlib import Path

ROOT = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, ROOT)

from _config import *
from doc_embedder.modules import *
from doc_embedder.functions import *


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
