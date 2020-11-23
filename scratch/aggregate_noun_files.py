import sys
from pathlib import Path

ROOT = str(Path(__file__).parent.parent.absolute())
print(ROOT)
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

review = 'book_nouns_agg.txt'
wiki = 'wiki_nouns.txt'
book_content = 'book_content_nouns.txt'

# fname = review
# fname = wiki
# fname = book_content

texts = b""
for fname in [review, wiki, book_content]:
    source_dir = '/home/ubuntu/yonghee/doc-embedder/book_nouns'
    path = os.path.join(source_dir, fname)
    with open(path, 'rb') as f:
        texts += '\n\n'.encode('utf8')
        texts += f.read()
        print(len(texts))


savefname = 'nouns_agg_all.txt'
savepath = os.path.join(source_dir, savefname)
with open(savepath, 'wb') as f:
    f.write(texts)
