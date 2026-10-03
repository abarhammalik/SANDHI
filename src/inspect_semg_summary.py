import pandas as pd
p = r'data\real\semg\bundle\full_indicators_summary.csv'
d = None
for enc in ('utf-8-sig', 'gbk', 'cp1252', 'latin-1'):
    try:
        d = pd.read_csv(p, encoding=enc)
        print('ENCODING OK:', enc)
        break
    except Exception as e:
        print('failed', enc, type(e).__name__)
print(d.shape)
print(list(d.columns))
print(d.head(3).T.to_string())
for c in d.columns:
    n = str(c).lower()
    if any(k in n for k in ('group', 'k-l', 'period', 'gender', 'age', 'womac', 'number')):
        print(c, '->', d[c].value_counts(dropna=False).head(10).to_dict())
