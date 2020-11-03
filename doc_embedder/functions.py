import sys
from pathlib import Path

ROOT = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, ROOT)

from .modules import *

def get_uniques_from_nested_lists(nested_lists):
    uniques = {}
    for one_line in nested_lists:
        for item in one_line:
            if not uniques.get(item):
                uniques[item] = 1
    return list(uniques.keys())


def get_item2idx(items, unique=False):
    item2idx, idx2item = dict(), dict()
    items_unique = items if unique else set(items)
    for idx, item in enumerate(items_unique):
        item2idx[item] = idx
        idx2item[idx] = item
    return item2idx, idx2item


def load_watcha_sample_corpus(n_sample=None):
    filepath = os.path.join(ROOT, 'data', 'tb_watcha_comments_top30.csv')
    df = pd.read_csv(filepath, header=0, index_col=0, encoding='utf8')
    df.dropna(axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    if isinstance(n_sample, int):
        if n_sample > 0:
            return df.sample(n=n_sample)
    return df
