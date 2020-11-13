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

model_dir = '/home/ubuntu/yonghee/doc-embedder/trained_models'

modelfname = 'w2v_1160957_50d_epoch99_loss0.model'
modelfname = 'w2v_1160957_100d_epoch99_loss0.model'
modelfname = 'w2v_1160957_200d_epoch99_loss0.model'

modelpath = os.path.join(model_dir, modelfname)
model2 = gensim.models.Word2Vec.load(modelpath)
# model2 = gensim.models.Word2Vec.load(modelpath, encoding='utf8')

print(model2.wv.most_similar(''))
print('hello')
print('asdfadsf')
