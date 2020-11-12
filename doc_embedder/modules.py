import sys
from pathlib import Path
import pandas as pd
import os
import re
from gensim import corpora, models
import logging
from pprint import pprint
from konlpy.tag import Mecab
from gensim.models.callbacks import CallbackAny2Vec
from time import perf_counter as now

ROOT = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, ROOT)


class callback(CallbackAny2Vec):
    def __init__(self, epochs=None, save_dir=None, testquery=None):
        self.epoch = epochs if epochs is not None else 0
        self.batch = 0
        self.t0 = None
        self.total_loss = 0
        self.current_loss = None
        self.save_dir = save_dir
        self.testquery = testquery if isinstance(testquery, str) else None

    def on_train_begin(self, model):
        print('start training')

    def on_epoch_begin(self, model):
        self.t0 = now()

    def on_batch_begin(self, model):
        print(f"epoch {self.epoch}, batch {self.batch}")
        self.batch += 1

    def on_epoch_end(self, model):
        dur = now() - self.t0
        loss = model.get_latest_training_loss()
        self.current_loss = loss if self.current_loss is None else loss - self.total_loss
        self.total_loss = loss

        timestr = f"{dur:.2f} sec" if dur < 120 else f"{dur / 60:.2f} min"
        timestr += ' elapsed'
        print(f"after epoch {self.epoch}, loss: {self.current_loss:.2f}, " + timestr)

        fname = f"w2v_{model.corpus_count}_{model.vector_size}d_epoch{self.epoch}"
        fname += f"_loss{self.current_loss:.0f}.model"
        modelpath = os.path.join(self.save_dir, fname)
        model.save(modelpath)

        self.epoch += 1
