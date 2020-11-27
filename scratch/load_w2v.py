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

modelpath = os.path.join(model_dir, modelfname200)
model200 = gensim.models.Word2Vec.load(modelpath)
model = model200
model.wv.most_similar('자동차')
model.wv.most_similar('런닝')
model.wv.most_similar('파리')
model.wv.most_similar(positive=['사람', '새'], negative=['집'])
model.wv.similarity('조깅', '러닝')

# SAVING DICTIONARY
save_dir = '/home/yonghee/yonghee/doc-embedder/trained_models'
modelfname200 = 'w2v_dict_2998865_200d_epoch99_loss0.pkl'
savepath = os.path.join(save_dir, modelfname200)
word2vec = {vocab: model.wv.vectors[model.wv.vocab[vocab].index] for vocab in model.wv.vocab.keys()}

with open(savepath, 'wb') as f:
    pickle.dump(word2vec, f)

print('hello')

# LOADING DICTIONARY
del word2vec
import os
import gensim
import pickle

save_dir = '/home/yonghee/yonghee/doc-embedder/trained_models'
modelfname200 = 'w2v_dict_2998865_200d_epoch99_loss0.pkl'
savepath = os.path.join(save_dir, modelfname200)
with open(savepath, 'rb') as f:
    w2v = pickle.load(f)
print(w2v.get("사랑")[:5])

model.wv.get_vector('사랑') == w2v.get("사랑")
