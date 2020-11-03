import sys
from pathlib import Path

ROOT = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, ROOT)

from doc_embedder.modules import *


def load_watcha_sample_corpus(n_sample=None):
    filepath = os.path.join(ROOT, 'data', 'tb_watcha_comments_top30.csv')
    df = pd.read_csv(filepath, header=0, index_col=0, encoding='utf8')
    df.dropna(axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    if isinstance(n_sample, int):
        if n_sample >= 0:
            return df.sample(n=n_sample)
    return df
