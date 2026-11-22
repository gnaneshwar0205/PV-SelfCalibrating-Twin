from common.simulate import run
import pandas as pd
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
for c in ['asmc','nnmpc']: run(c)
rows=[]
for c in ['asmc','nnmpc']:
    rows.append(json.loads((ROOT/'results'/c/'summary.json').read_text()))
pd.DataFrame(rows).to_csv(ROOT/'results'/'comparison'/'summary_comparison.csv',index=False)
print('Saved comparison to results/comparison/summary_comparison.csv')
