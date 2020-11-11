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

source_dir = '/home/ubuntu/yonghee/doc-embedder/book_nouns'
filelist = [filename for filename in os.listdir(source_dir) if filename.startswith("corpora_")]

filename = filelist[1]
filepath = os.path.join(source_dir, filename)
with open(filepath, 'rb') as f:
    raw = f.read().decode("utf8")
lines = raw.split("\n\n")
titles, texts = lines[0][:-2], lines[1:]
titles = titles.split('\t\t')

MIN_TOKENS = 5
tokens_matrix = []
for text in texts:
    spl = text.split(" ")
    if len(spl) > MIN_TOKENS:
        tokens_matrix.append(spl)

model = gensim.models.Word2Vec(sentences=tokens_matrix, size=50, window=5, sg=1, hs=0, )

# filelist = ['corpora_mecab_938.pk']
for idx, filename in enumerate(filelist):
    print(f"{idx}/{len(filelist)} {filename}")
    t0 = now()
    make_nouns_file(filename)
    dur = now() - t0
    print(f"{idx}/{len(filelist)} {filename} elapsed {dur / 60:.6f} min")
