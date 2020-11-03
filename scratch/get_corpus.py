import sys
from pathlib import Path

ROOT = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, ROOT)

from modules import *

filepath = os.path.join(ROOT, 'data', 'tb_watcha_comments_top30.csv')
df = pd.read_csv(filepath, header=0, index_col=0, encoding='utf8')
print(df.head())
df = df.dropna(axis=0, inplace=False)
df.reset_index(drop=True, inplace=True)

df.groupby(by='title', axis=0).count().sort_values(by='text', ascending=False)

df.sample(n=50)