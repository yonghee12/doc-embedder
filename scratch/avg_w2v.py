import sys
from pathlib import Path

ROOT = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, ROOT)

from itertools import chain

from _config import *
from doc_embedder.modules import *
from doc_embedder.functions import *

from biblyhouse.base import BookHandlerBM

r = DirectRedis(**BOOK_ID_MAPPING)
r_group = DirectRedis(**GROUP_ID_MAPPING)


def get_book_name_from_book_id(book_id):
    return r.hget("book_id", f'{book_id}:book_name')


def get_book_id_from_group_id(group_id):
    # group_id = 'S4EDD961D81F6BF04'
    group = r_group.lrange(group_id, start=0, end=0)
    if group:
        book_id = group[0]
        firstcontent = r_group.hget('firstcontent', book_id)
        return firstcontent if firstcontent else book_id
    else:
        return None


def get_book_name_from_book_index(index):
    """
    get book name from book index which can be book_id or group_id
    """
    book_id = get_book_id_from_group_id(index)
    book_id = book_id if book_id is not None else index
    return get_book_name_from_book_id(book_id)


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
        'corpora_mecab_925_r.txt'
    ]

    def __init__(self, source_dir, nickname, from_pickle=False, overwrite_pickle=False, model=None, prefix=None,
                 suffix=None, source='bestseller'):
        """
        :param source_dir: 'all', 'sample', 'bestseller'
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
        self.bestsellers = self._get_bestseller_set() if self.use_bestseller else None
        self.filelist = self.samples if source == 'sample' else self._get_filelist(prefix, suffix)
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

    def _get_filelist(self, prefix, suffix):
        prefix = 'corpora_mecab'
        suffix = '_r.txt'
        if prefix or suffix:
            if not prefix:
                return [fname for fname in os.listdir(self.source_dir) if fname.endswith(str(suffix))]
            elif not suffix:
                return [fname for fname in os.listdir(self.source_dir) if fname.startswith(str(prefix))]
            else:
                return [fname for fname in os.listdir(self.source_dir)
                        if fname.startswith(str(prefix)) and fname.endswith(str(suffix))]
        else:
            return os.listdir(self.source_dir)

    def _get_bestseller_set(self):
        bh = BookHandlerBM()
        bestseller_path = os.path.join(ROOT, "bestseller.csv")
        bestseller = pd.read_csv(bestseller_path, header=None)
        bestseller_book_ids = set(bestseller.iloc[:, 0])
        print('making bestseller group_id index...')
        for book_id in tqdm(list(bestseller_book_ids)):
            bestseller_book_ids.add(bh.get_recom_id(book_id))
        self.doc2bid = {title: bid for title, bid in zip(bestseller.iloc[:, 2], bestseller.iloc[:, 0])}
        return bestseller_book_ids

    def _update_corpora(self):
        for idx, filename in enumerate(self.filelist):
            print(f"{idx + 1}/{len(self.filelist)} {filename}")
            t0 = now()
            self._update_corpora_by_filename(filename)
            dur = now() - t0
            printstr = f"{idx + 1}/{len(self.filelist)} {filename}"
            printstr += f" {dur / 60:.6f} min" if dur > 120 else f" {dur:.6f} sec"
            print(printstr + f" elapsed, total keys: {len(self.corpora)}")

        for key in self.corpora.keys():
            self.corpora[key] = list(chain.from_iterable([line.split(" ") for line in self.corpora[key]]))

        if self.overwrite_pickle:
            with open(self.doc_savepath, 'wb') as f:
                print('writing pickle file...')
                pickle.dump(self.corpora, f)

    def _update_corpora_by_filename(self, filename):
        filepath = os.path.join(self.source_dir, filename)

        with open(filepath, 'rb') as f:
            raw = f.read().decode("utf8")
        lines = raw.split("\n\n")
        indices = lines[0].split('\t\t')[:-1]
        texts = lines[1:]

        for index, text in tqdm(zip(indices, texts)):
            if self.use_bestseller and index not in self.bestsellers:
                continue

            spl = text.split(" ")
            if len(spl) > MIN_TOKENS:
                tokens = ' '.join(spl)
                if index in self.corpora:
                    self.corpora[index].add(tokens)
                else:
                    self.corpora[index] = {tokens}

    def make_document_vector_matrix(self, save=False):
        wv_mat = []
        self.idx2doc = []
        self.doc2idx = {}
        for book_id, tokens in tqdm(self.corpora.items()):
            vector = self.get_vectors_from_tokens(tokens, pooling='average')
            if vector is None:
                raise Exception("no vector")
            wv_mat.append(vector)
            self.idx2doc.append(book_id)
            self.doc2idx[book_id] = len(self.idx2doc) - 1

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

    def get_most_similar(self, doc_id, top=5):
        vector = self.similarity_matrix[doc_id]
        top_indices = np.argpartition(-vector, top + 1)[:top + 1]
        sorted_top_indices = top_indices[np.argsort(-vector[top_indices])]
        sorted_top_indices = sorted_top_indices[1:]
        return list(zip([self.idx2doc[i] for i in sorted_top_indices], vector[sorted_top_indices]))

    def get_most_similars_matrix(self, top_n=10):
        self.most_similars = {}
        for i in range(len(self.idx2doc)):
            book_index = self.idx2doc[i]
            self.most_similars[book_index] = self.get_most_similar(i, top_n)
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
    MAKE_DOCVEC = False

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
    print(model.wv.most_similar('사랑'))
    # model = None

    corpora = VectorizeCorpora(source_dir,
                               nickname='rep_all_200d',
                               from_pickle=True,
                               overwrite_pickle=False,
                               model=model,
                               prefix='corpora_mecab',
                               suffix='_r.txt',
                               source='bestseller')

    if MAKE_DOCVEC:
        print('making document vectors.. ')
        corpora.make_document_vector_matrix(save=True)
        corpora.make_similarity_matrix(save=True)
    else:
        corpora.load_averaged_docvec_matrix()
        corpora.load_document_similarity_matrix()

    print(corpora.get_most_similar(0, 5))
    sims = corpora.get_most_similars_matrix()
    name_func = get_book_name_from_book_index

    df = pd.DataFrame(sims)
    print(df.head())

    df.to_csv(os.path.join(ROOT, 'data', 'book_sims_book_id.csv'), encoding='utf8')
    df.T.to_csv(os.path.join(ROOT, 'data', 'book_sims_book_id_T.csv'), encoding='utf8')

    # save most_similars
    path = os.path.join(ROOT, 'data', 'bestseller_most_similars.pkl')
    with open(path, 'wb') as f:
        pickle.dump(sims, f)

    BOOK_NAME = True
    if BOOK_NAME:
        sims_name = {name_func(k): [(name_func(i), s,) for i, s in v] for k, v in sims.items()}

    df = pd.DataFrame(sims_name)
    print(df.head())
    df.to_csv(os.path.join(ROOT, 'data', 'book_sims_bookname.csv'), encoding='utf8')
    df.T.to_csv(os.path.join(ROOT, 'data', 'book_sims_bookname_T.csv'), encoding='utf8')

    # load to test
    # import pickle
    # with open(path, 'rb') as f:
    #     sims_load = pickle.load(f)
