import sys
from pathlib import Path

ROOT = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, ROOT)

import pickle
import pandas as pd
import numpy as np
import gensim.models
from gensim.models.callbacks import CallbackAny2Vec
from string import punctuation
from time import perf_counter as now
from direct_redis import DirectRedis
from tqdm import tqdm
from progress_timer import Timer

from _config import *
from doc_embedder.modules import *
from doc_embedder.functions import *

r = DirectRedis(**BOOK_ID_MAPPING)

source_dir = '/home/ubuntu/yonghee/doc-embedder/book_nouns'
agg_filename = 'book_nouns_agg.txt'
agg_filepath = os.path.join(source_dir, agg_filename)

agg_filename_sample = 'book_nouns_agg_sample.txt'
agg_filepath_sample = os.path.join(source_dir, agg_filename_sample)


def make_sample(max_lines, original_path, sample_path):
    with open(sample_path, 'w') as f:
        f.write('')

    sample = ''
    with open(original_path, 'rb') as f:
        i = 0
        timer = Timer(max_lines)
        for line in f:
            timer.time_check(i)
            sample += line.decode('utf8')
            i += 1
            if len(sample) >= 10 ** 6 or i >= max_lines:
                with open(sample_path, 'ab') as f:
                    f.write(sample.encode('utf8'))
                sample = ''
                if i >= max_lines:
                    return None


model_dir = '/home/ubuntu/yonghee/doc-embedder/trained_models'
# make_sample(10000, agg_filepath, agg_filepath_sample)
print('start training')
for vector_size in [50, 100, 200]:
    corpus = agg_filepath
    # corpus = agg_filepath_sample
    model_config = {
        'corpus_file': corpus,
        'size': vector_size,
        'window': 5,
        "sg": 0,
        "hs": 0,
        "iter": 0,
        # "callbacks": [callback()],
        'callbacks': [callback(save_dir=model_dir)],
        "compute_loss": True,
        "workers": 24
    }
    model = gensim.models.Word2Vec(**model_config, seed=42)
    print(model.wv.most_similar('사랑'))

    train_config = {
        'corpus_file': corpus,
        # 'callbacks': [callback(save_dir=model_dir, testquery='사랑')],
        'callbacks': [callback(save_dir=model_dir)],
        'compute_loss': True,
        'report_delay': 0.1,
        'total_examples': model.corpus_count,
        'total_words': model.corpus_total_words,
        'epochs': 100,
    }

    model.train(**train_config)
    print(model.wv.most_similar('사랑'))

# model_dir = '/home/ubuntu/yonghee/doc-embedder/trained_models'
# modelfname = 'w2v_20031_50d_epoch2_loss406002.model'
# modelpath = os.path.join(model_dir, modelfname)
# model2 = gensim.models.Word2Vec.load(modelpath)
# model2 = gensim.models.Word2Vec.load(modelpath, encoding='utf8')
# print(model2.wv.most_similar('사랑'))
print('hello')
print('asdfadsf')
