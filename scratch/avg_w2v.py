import sys
from pathlib import Path

ROOT = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, ROOT)

from itertools import chain

from _config import *
from doc_embedder.modules import *
from doc_embedder.functions import *

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

    def __init__(self, source_dir, nickname, from_pickle=False, overwrite_pickle=False, model=None, prefix=None,
                 source='bestseller'):
        """
        :param source_dir: 'all', 'sample', 'bestseller'
        :param model:
        :param prefix:
        :param source:
        """
        self.source_dir = source_dir
        self.nickname = nickname
        self.doc_savepath = os.path.join(source_dir, f'{nickname}_corpora_processed.pkl')
        self.avgvec_savepath = os.path.join(source_dir, f'{nickname}_document_vectors.npy')
        self.simindex_savepath = os.path.join(source_dir, f'{nickname}_similarity_index_matrix.npy')
        self.idx2doc_savepath = os.path.join(source_dir, f'{nickname}_idx2doc.pkl')
        self.doc2idx_savepath = os.path.join(source_dir, f'{nickname}_doc2idx.pkl')
        self.overwrite_pickle = overwrite_pickle if not from_pickle else False
        self.use_bestseller = True if source == 'bestseller' else False
        self.doc2bid = None
        self.bestsellers = self._get_bestseller_list() if self.use_bestseller else None
        self.filelist = self.samples if source == 'sample' else self._get_filelist(prefix)
        self.corpora = dict()
        self.word2idx = {vocab: model.wv.vocab[vocab].index for vocab in model.wv.vocab.keys()}
        self.vectors = model.wv.vectors[:]
        self.wv_mat = None
        self.idx2doc = None
        self.doc2idx = None
        self.norms = None
        self.similarity_matrix = None

        if from_pickle:
            print('Reading Pickle File..')
            t0 = now()
            with open(self.doc_savepath, 'rb') as f:
                self.corpora = pickle.load(f)
            print(f"Reading pickle took {(now() - t0) / 60:.2f} min")
        else:
            self._update_corpora()

    def upload_model(self, model):
        self.model = model

    def load_averaged_docvec_matrix(self):
        self.wv_mat = np.load(self.avgvec_savepath, fix_imports=False)
        with open(self.idx2doc_savepath, 'rb') as f:
            self.idx2doc = pickle.load(f)
        with open(self.doc2idx_savepath, 'rb') as f:
            self.doc2idx = pickle.load(f)

    def load_document_similarity_matrix(self):
        self.similarity_matrix = np.load(self.simindex_savepath, fix_imports=False)

    def _get_filelist(self, prefix):
        if prefix:
            return [filename for filename in os.listdir(self.source_dir)
                    if filename.startswith(f"{prefix}")]
        else:
            return [filename for filename in os.listdir(self.source_dir)]

    def _get_bestseller_list(self):
        bestseller_path = os.path.join(ROOT, "bestseller.csv")
        bestseller = pd.read_csv(bestseller_path, header=None)
        self.doc2bid = {title: bid for title, bid in zip(bestseller.iloc[:, 2], bestseller.iloc[:, 0])}
        return bestseller.iloc[:, 2].to_list()

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

        if self.overwrite_pickle:
            with open(self.doc_savepath, 'wb') as f:
                pickle.dump(self.corpora, f)

    def _update_corpora_by_filename(self, filename):
        filepath = os.path.join(self.source_dir, filename)

        with open(filepath, 'rb') as f:
            raw = f.read().decode("utf8")
        lines = raw.split("\n\n")
        titles = lines[0].split('\t\t')[:-1]
        texts = lines[1:]

        for title, text in tqdm(zip(titles, texts)):
            if self.use_bestseller and title not in self.bestsellers:
                continue

            spl = text.split(" ")
            if len(spl) > MIN_TOKENS:
                tokens = ' '.join(spl)
                if title in self.corpora:
                    self.corpora[title].add(tokens)
                else:
                    self.corpora[title] = {tokens}

    def make_document_vector_matrix(self, save=False):
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

        if save:
            np.save(self.avgvec_savepath, self.wv_mat, fix_imports=False)
            with open(self.idx2doc_savepath, 'wb') as f:
                pickle.dump(self.idx2doc, f)
            with open(self.doc2idx_savepath, 'wb') as f:
                pickle.dump(self.doc2idx, f)

        return f"Total {len(self.idx2doc)} documents indexed."

    def make_similarity_matrix(self, save=True):
        self.norms = np.linalg.norm(self.wv_mat, axis=1, keepdims=True)
        self.similarity_matrix = (self.wv_mat @ self.wv_mat.T) / (self.norms @ self.norms.T)
        if save:
            np.save(self.simindex_savepath, self.similarity_matrix, fix_imports=False)

    def get_most_similar(self, doc_id, top=5, return_book_id=False):
        vector = self.similarity_matrix[doc_id]
        top_indices = np.argpartition(-vector, top + 1)[:top + 1]
        sorted_top_indices = top_indices[np.argsort(-vector[top_indices])]
        sorted_top_indices = sorted_top_indices[1:]
        if return_book_id:
            return list(zip([self.doc2bid[self.idx2doc[i]] for i in sorted_top_indices], vector[sorted_top_indices]))
        else:
            return list(zip([self.idx2doc[i] for i in sorted_top_indices], vector[sorted_top_indices]))

    def get_most_similars_matrix(self, return_book_id=False):
        self.most_similars = {}
        for i in range(len(self.idx2doc)):
            book_id = self.doc2bid[self.idx2doc[i]]
            self.most_similars[book_id] = self.get_most_similar(i, 10, return_book_id)
        return self.most_similars

    def get_vectors_from_tokens(self, tokens, pooling=None):
        vectors = []
        for token in tokens:
            vector = self.get_vector(token)
            if vector is None:
                continue
            vectors.append(vector)

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


if __name__ == '__main__':
    MAKE_DOCVEC = True

    MIN_TOKENS = 5
    source_dir = '/home/yonghee/yonghee/doc-embedder/book_nouns'

    models = {
        'review': {
            'dir': '/home/yonghee/yonghee/doc-embedder/trained_models',
            '50d': '50d/w2v_1160957_50d_epoch99_loss0.model',
            '100d': '100d/w2v_1160957_100d_epoch99_loss0.model',
            '200d': '200d/w2v_1160957_200d_epoch99_loss0.model'
        },
        'review_content_wiki': {
            'dir': '/home/yonghee/yonghee/doc-embedder/trained_models/review_content_wiki',
            '50d': '50d/w2v_2998865_50d_epoch99_loss0.model',
            '100d': '100d/w2v_2998865_100d_epoch99_loss0.model',
            '200d': '200d/w2v_2998865_200d_epoch99_loss0.model'
        }
    }
    model_info = models['review_content_wiki']

    modelpath = os.path.join(model_info['dir'], model_info['200d'])
    model = gensim.models.Word2Vec.load(modelpath)
    # model = None

    corpora = VectorizeCorpora(source_dir, 'all_200d', from_pickle=False, overwrite_pickle=True, model=model,
                               prefix='corpora_mecab',
                               source='bestseller')

    if MAKE_DOCVEC:
        corpora.make_document_vector_matrix(save=True)
        corpora.make_similarity_matrix(save=True)
    else:
        corpora.load_averaged_docvec_matrix()
        corpora.load_document_similarity_matrix()

    print(corpora.get_most_similar(0, 5))
    sims = corpora.get_most_similars_matrix(return_book_id=False)

    # save most_similars
    path = os.path.join(ROOT, 'data', 'bestseller_most_similars_bookname.pkl')
    with open(path, 'wb') as f:
        pickle.dump(sims, f)

    del sims
    sims = corpora.get_most_similars_matrix(return_book_id=True)

    # save most_similars
    path = os.path.join(ROOT, 'data', 'bestseller_most_similars.pkl')
    with open(path, 'wb') as f:
        pickle.dump(sims, f)

    # load to test
    import pickle
    with open(path, 'rb') as f:
        sims_load = pickle.load(f)

