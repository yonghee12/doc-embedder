import sys
from pathlib import Path

ROOT = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, ROOT)

import pickle
import pandas as pd
import numpy as np
import gensim.models
from string import punctuation
from time import perf_counter as now
from direct_redis import DirectRedis
from tqdm import tqdm

from _config import *
from doc_embedder.modules import *
from doc_embedder.functions import *

r = DirectRedis(**BOOK_ID_MAPPING)


def get_book_name_from_book_id(book_id):
    return r.hget("book_id", f'{book_id}:book_name')


source_dir = '/corpora/book_content'
filelist = [filename for filename in os.listdir(source_dir)
            if filename.startswith("corpora_") and filename.endswith('_r.pk') and not filename.endswith("r_r.pk")]

allowed_pos = {
    'NNG',
    'NNP',
}


def make_nouns_file(filename, return_book_name=False):
    path = os.path.join(source_dir, filename)
    corpora = pd.read_pickle(path)

    book_index_agg, nouns_agg = '', ''
    for corpus in tqdm(corpora):
        book_id, rep_id, tokens = corpus
        book_index = get_book_name_from_book_id(book_id) if return_book_name else rep_id
        nouns = [tok for tok, pos in tokens if pos in allowed_pos]

        if nouns:
            book_index_agg += str(book_index) + '\t\t'
            nouns_agg += ' '.join(nouns) + '\n\n'

    if not nouns_agg:
        return None

    nouns_agg = nouns_agg[:-2]  # 마지막 \n\n 제거

    filename_nouns = filename.split(".")[0] + '.txt'
    save_dir = '/home/yonghee/yonghee/doc-embedder/book_nouns'
    savepath = os.path.join(save_dir, filename_nouns)

    with open(savepath, 'wb') as f:
        write = book_index_agg + '\n\n' + nouns_agg
        f.write(write.encode('utf8'))


if __name__ == '__main__':
    BOOK_NAME = False
    # filelist = [
    #     'corpora_mecab_881_r.pk',
    #     'corpora_mecab_87_r.pk',
    #     'corpora_mecab_925_r.pk',
    #     'corpora_mecab_911_r.pk',
    # ]

    for idx, filename in enumerate(filelist):
        print(f"{idx + 1}/{len(filelist)} {filename}")
        t0 = now()
        make_nouns_file(filename, return_book_name=BOOK_NAME)
        dur = now() - t0
        print(f"{idx + 1}/{len(filelist)} {filename} elapsed {dur / 60:.6f} min")

    print('hello')
