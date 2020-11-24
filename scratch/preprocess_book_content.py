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

allowed_pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣]+')
compress_punct = re.compile("([.?!])\1+")

MIN_TOKENS = 5

source_dir = '/home/ubuntu/yonghee/doc-embedder/data'
filename = 'book_content.pk'
path = os.path.join(source_dir, filename)

with open(path, 'rb') as f:
    texts = pickle.load(f)
    # sample = pickle.load(f)[:100]
# texts = deepcopy(sample)
texts = list(chain.from_iterable(texts))
texts = [compress_punct.sub(r'\1', line) for line in texts]
texts = [allowed_pattern.sub(' ', line).strip() for line in texts]
print(texts[0])

allowed_pos = {'NNG', 'NNP', }

tok = Mecab()
nouns_agg = []
for line in tqdm(texts):
    tagged = tok.pos(line)
    nouns = [word for word, pos in tagged if pos in allowed_pos]
    if len(nouns) > MIN_TOKENS:
        nouns_agg.append(' '.join(nouns))

filename_nouns = 'book_content_nouns.txt'
save_dir = '/home/ubuntu/yonghee/doc-embedder/book_nouns'
savepath = os.path.join(save_dir, filename_nouns)

with open(savepath, 'wb') as f:
    f.write('\n\n'.join(nouns_agg).encode('utf8'))
