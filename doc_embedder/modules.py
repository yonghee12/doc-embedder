import sys
from pathlib import Path
import pandas as pd
import os
import re
from gensim import corpora, models
import logging
from pprint import pprint
from konlpy.tag import Mecab

ROOT = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, ROOT)
