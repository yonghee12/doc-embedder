import sys
from pathlib import Path

ROOT = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, ROOT)

import pickle
import numpy as np
from time import perf_counter as now
from direct_redis import DirectRedis
from tqdm import tqdm
from progress_timer import Timer

from _config import *
from doc_embedder.modules import *
from doc_embedder.functions import *

r = DirectRedis(**WORD_VECTORS)

from doc_embedder.modules import *

model_dir = '/home/ubuntu/yonghee/doc-embedder/trained_models'
modelfname = 'w2v_1160957_200d_epoch99_loss0.model'
modelpath = os.path.join(model_dir, modelfname)
model = models.Word2Vec.load(modelpath)


def get_vector(word):
    return model.wv.vectors[model.wv.vocab[word].index]


def get_vector_safe(word):
    if word in model.wv.vocab:
        return model.wv.vectors[model.wv.vocab[word].index]
    else:
        return None


# TODO: vector -> map -> redis 효율 로직
for key in model.wv.vocab.keys():
    pass
