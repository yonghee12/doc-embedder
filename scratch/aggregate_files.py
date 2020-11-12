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
with open(final_filepath, 'w') as final:
    final.write('')


def make_aggregated_file(filename):
    filepath = os.path.join(source_dir, filename)

    with open(filepath, 'rb') as f:
        raw = f.read().decode("utf8")
    lines = raw.split("\n\n")
    texts = lines[1:]

    tokens_agg = set()
    for text in tqdm(texts):
        spl = text.split(" ")
        if len(spl) > MIN_TOKENS:
            spl = ' '.join(spl)
            tokens_agg.add(spl)

    tokens_agg = '\n'.join(tokens_agg)
    with open(final_filepath, 'ab') as final:
        print(tokens_agg[:200])
        final.write(tokens_agg.encode('utf8'))


for idx, filename in enumerate(filelist):
    print(f"{idx + 1}/{len(filelist)} {filename}")
    t0 = now()
    make_aggregated_file(filename)
    dur = now() - t0
    print(f"{idx + 1}/{len(filelist)} {filename} elapsed {dur / 60:.6f} min")
