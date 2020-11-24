#!/home/ubuntu/.yonghee_env/bin/python3
# -*- coding: utf-8 -*-

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


source_dir = '/home/ubuntu/yonghee/doc-embedder/book_nouns'
agg_filename_all = 'nouns_agg_all.txt'
agg_filepath_all = os.path.join(source_dir, agg_filename_all)

agg_filename = 'book_nouns_agg.txt'
agg_filepath = os.path.join(source_dir, agg_filename)

agg_filename_sample = 'book_nouns_agg_sample.txt'
agg_filepath_sample = os.path.join(source_dir, agg_filename_sample)

model_dir_parent = '/home/ubuntu/yonghee/doc-embedder/trained_models/review_content_wiki'
# make_sample(10000, agg_filepath, agg_filepath_sample)
epochs = 100
print('start training')
for vector_size in [50, 100, 200]:
    corpus = agg_filepath_all

    model_dir = model_dir_parent + f'/{vector_size}d'
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
    print(f'{vector_size}d building vocab...')
    t0 = now()
    model = gensim.models.Word2Vec(**model_config, seed=42)
    dur = now() - t0
    print(f"Building took {dur:.2f} sec, {dur / 60:.2f} min")

    try:
        print(model.wv.most_similar('사랑'))
    except Exception as e:
        print(str(e))

    train_config = {
        'corpus_file': corpus,
        # 'callbacks': [callback(save_dir=model_dir, testquery='사랑')],
        'callbacks': [callback(save_dir=model_dir)],
        'compute_loss': True,
        'report_delay': 0.1,
        'total_examples': model.corpus_count,
        'total_words': model.corpus_total_words,
        'epochs': epochs,
    }

    print(f'{vector_size}d training start...')
    t0 = now()
    model.train(**train_config)
    dur = now() - t0
    print(f"training {epochs} epochs took {dur:.2f} sec, {dur / 60:.2f} min")

    try:
        print(model.wv.most_similar('사랑'))
    except Exception as e:
        print(str(e))

# model_dir = '/home/ubuntu/yonghee/doc-embedder/trained_models'
# modelfname = 'w2v_20031_50d_epoch2_loss406002.model'
# modelpath = os.path.join(model_dir, modelfname)
# model2 = gensim.models.Word2Vec.load(modelpath)
# model2 = gensim.models.Word2Vec.load(modelpath, encoding='utf8')
# print(model2.wv.most_similar('사랑'))
print('hello')
print('asdfadsf')
