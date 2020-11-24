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
from konlpy.tag import Mecab
from itertools import chain
from copy import deepcopy

MIN_TOKENS = 5

source_dir = '/home/ubuntu/yonghee/doc-embedder/data'
filename = 'wiki.txt'
path = os.path.join(source_dir, filename)

with open(path, 'rb') as f:
    texts = f.readlines()
    # sample = f.readlines()[:100]

allowed_pos = {'NNG', 'NNP', }
nouns_agg = []
for text in tqdm(texts):
    line = text.decode('utf8').strip()
    tokens = list(map(lambda x: x.split('/'), line.split(" ")))
    tokens = [token for token in tokens if len(token) == 2]
    nouns = [word for word, pos in tokens if pos in allowed_pos]
    if len(nouns) > MIN_TOKENS:
        nouns_agg.append(' '.join(nouns))

filename_nouns = 'wiki_nouns.txt'
save_dir = '/home/ubuntu/yonghee/doc-embedder/book_nouns'
savepath = os.path.join(save_dir, filename_nouns)

with open(savepath, 'wb') as f:
    f.write('\n\n'.join(nouns_agg).encode('utf8'))
