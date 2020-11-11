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


source_dir = '/home/ubuntu/corpora/book_content'
filelist = [filename for filename in os.listdir(source_dir) if filename.startswith("corpora_")]

allowed_pos = {
    'NNG',
    'NNP',
}


def make_nouns_file(filename):
    path = os.path.join(source_dir, filename)
    corpora = pd.read_pickle(path)

    book_name_agg, nouns_agg = '', ''
    for corpus in tqdm(corpora):
        bid, tokens = corpus
        book_name = get_book_name_from_book_id(bid)
        nouns = [tok for tok, pos in tokens if pos in allowed_pos]

        if nouns:
            book_name_agg += str(book_name) + '\t\t'
            nouns_agg += ' '.join(nouns) + '\n\n'

    if not nouns_agg:
        return None

    nouns_agg = nouns_agg[:-2]

    filename_nouns = filename.split(".")[0] + '.txt'
    save_dir = '/home/ubuntu/yonghee/doc-embedder/book_nouns'
    savepath = os.path.join(save_dir, filename_nouns)

    with open(savepath, 'wb') as f:
        write = book_name_agg + '\n\n' + nouns_agg
        f.write(write.encode('utf8'))


# filelist = ['corpora_mecab_938.pk']
for idx, filename in enumerate(filelist):
    print(f"{idx}/{len(filelist)} {filename}")
    t0 = now()
    make_nouns_file(filename)
    dur = now() - t0
    print(f"{idx}/{len(filelist)} {filename} elapsed {dur / 60:.6f} min")
