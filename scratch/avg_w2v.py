import sys
from pathlib import Path

ROOT = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, ROOT)

from itertools import chain

from _config import *
from doc_embedder.modules import *
from doc_embedder.functions import *

MIN_TOKENS = 5
source_dir = '/home/ubuntu/yonghee/doc-embedder/book_nouns'


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


class VectorizeCorpora:
    samples = [
        'corpora_mecab_880.txt',
        'corpora_mecab_938.txt',
        'corpora_mecab_97.txt',
        'corpora_mecab_910.txt',
        'corpora_mecab_936.txt',
        'corpora_mecab_80.txt'
        # 'corpora_mecab_884.txt'
    ]

    def __init__(self, source_dir, model=None, prefix=None, use_sample=False):
        self.source_dir = source_dir
        self.filelist = self._get_filelist(prefix) if not use_sample else self.samples
        self.corpora = dict()
        self.word2idx = {vocab: model.wv.vocab[vocab].index for vocab in model.wv.vocab.keys()}
        self.vectors = model.wv.vectors[:]
        self._update_corpora()
        self.wv_mat = None
        self.idx2doc = None
        self.doc2idx = None
        self.norms = None
        self.similarity_matrix = None

    def upload_model(self, model):
        self.model = model

    def _get_filelist(self, prefix):
        if prefix:
            return [filename for filename in os.listdir(self.source_dir)
                    if filename.startswith(f"{prefix}")]
        else:
            return [filename for filename in os.listdir(self.source_dir)]

    def _update_corpora(self):
        for idx, filename in enumerate(self.filelist):
            print(f"{idx + 1}/{len(self.filelist)} {filename}")
            t0 = now()
            self._update_corpora_by_filename(filename)
            dur = now() - t0
            printstr = f"{idx + 1}/{len(self.filelist)} {filename}"
            printstr += f" {dur / 60:.6f} min" if dur > 120 else f" {dur:.6f} sec"
            print(printstr + " elapsed")

        for key in self.corpora.keys():
            self.corpora[key] = list(chain.from_iterable([line.split(" ") for line in self.corpora[key]]))

    def _update_corpora_by_filename(self, filename):
        filepath = os.path.join(self.source_dir, filename)

        with open(filepath, 'rb') as f:
            raw = f.read().decode("utf8")
        lines = raw.split("\n\n")
        titles = lines[0].split('\t\t')[:-1]
        texts = lines[1:]

        for title, text in tqdm(zip(titles, texts)):
            spl = text.split(" ")
            if len(spl) > MIN_TOKENS:
                tokens = ' '.join(spl)
                if title in self.corpora:
                    self.corpora[title].add(tokens)
                else:
                    self.corpora[title] = {tokens}

    def make_document_vector_matrix(self):
        wv_mat = []
        self.idx2doc = []
        self.doc2idx = {}
        for title, tokens in tqdm(self.corpora.items()):
            vector = self.get_vectors_from_tokens(tokens, pooling='average')
            if vector is None:
                raise Exception("no vector")
            wv_mat.append(vector)
            self.idx2doc.append(title)
            self.doc2idx[title] = len(self.idx2doc) - 1

        self.wv_mat = np.array(wv_mat)
        return f"Total {len(self.idx2doc)} documents indexed."

    def make_similarity_matrix(self):
        self.norms = np.linalg.norm(self.wv_mat, axis=1, keepdims=True)
        self.similarity_matrix = (self.wv_mat @ self.wv_mat.T) / (self.norms @ self.norms.T)

    def get_most_similar(self, doc_id):
        vector = self.similarity_matrix[doc_id]
        top_indices = np.argpartition(-vector, 5)[:5]
        sorted_top_indices = top_indices[np.argsort(-vector[top_indices])]
        list(zip([self.idx2doc[i] for i in sorted_top_indices], vector[sorted_top_indices]))
        return list(zip([self.idx2doc[i] for i in sorted_top_indices], vector[sorted_top_indices]))

        for i in range(20, 40):
            vector = self.similarity_matrix[i]
            top_indices = np.argpartition(-vector, 5)[:5]
            sorted_top_indices = top_indices[np.argsort(-vector[top_indices])]
            print(list(zip([self.idx2doc[i] for i in sorted_top_indices], vector[sorted_top_indices])))

    def get_vectors_from_tokens(self, tokens, pooling=None):
        vectors = []
        for token in tokens:
            vector = self.get_vector(token)
            if vector is None:
                continue
            vectors.append(vector)
        np.array(vectors).mean(axis=0)

        if pooling == 'average' or pooling == 'avg':
            if not vectors:
                return None
            return np.array(vectors).mean(axis=0)

        return vectors

    def get_vector(self, word):
        if word in self.word2idx:
            return self.vectors[self.word2idx[word]]
        else:
            return None


model_dir = '/home/ubuntu/yonghee/doc-embedder/trained_models'

# modelfname = 'w2v_1160957_50d_epoch99_loss0.model'
modelfname = 'w2v_1160957_100d_epoch99_loss0.model'
# modelfname = 'w2v_1160957_200d_epoch99_loss0.model'

modelpath = os.path.join(model_dir, modelfname)
model = gensim.models.Word2Vec.load(modelpath)
#
# word = '책임'
# if word in word2idx:
#     print(vectors[word2idx[word]])

updater = VectorizeCorpora(source_dir, model=model, prefix='corpora', use_sample=True)
updater.make_document_vector_matrix()
updater.make_similarity_matrix()
updater.get_most_similar(1)
# with open(os.path.join(ROOT, 'updater.pkl'), 'wb') as f:
#     pickle.dump(updater, f)

r = DirectRedis(**BOOK_ID_MAPPING)
len(updater.corpora)

# wv_mat = np.random.random((5, 20))
# cos_sim
df = pd.DataFrame(cos_sim, columns=corpus_agg.index, index=corpus_agg.index)
corpus_agg.index
print(df)
filename = "movie_agg.csv"
path = os.path.join(ROOT, filename)
df.to_csv(path, encoding='utf8')
print('hellaosdf')
