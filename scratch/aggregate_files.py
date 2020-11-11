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

MIN_TOKENS = 5

source_dir = '/home/ubuntu/yonghee/doc-embedder/book_nouns'
filelist = [filename for filename in os.listdir(source_dir) if filename.startswith("corpora_")]

final_filename = 'book_nouns_agg.txt'
final_filepath = os.path.join(source_dir, final_filename)
# ff = open(final_filename, 'wb')
ff = open(final_filepath, 'w')
ff.write("hello\n\n")
ff.close()
ff = open(final_filepath, 'a')
ff.write("world\n\n")
ff.close()

filelist = filelist[:2]
for filename in filelist:
    filepath = os.path.join(source_dir, filename)

    with open(filepath, 'rb') as f:
        raw = f.read().decode("utf8")
    lines = raw.split("\n\n")
    texts = lines[1:]

    tokens_matrix = []
    for text in texts:
        spl = text.split(" ")
        if len(spl) > MIN_TOKENS:
            spl = ' '.join(spl)
            tokens_matrix.append(spl)
