import json, glob, os
import pandas as pd
base = r'data\real\imu\dataset\data'
rows = []
for f in glob.glob(os.path.join(base, '*', '*', '*', '*', '*_meta.json')):
    m = json.load(open(f))
    r = {k: v for k, v in m.items() if 'Events' not in k and k != 'uturnBoundaries'}
    r['n_left_events'] = len(m.get('leftGaitEvents', []))
    r['n_right_events'] = len(m.get('rightGaitEvents', []))
    r['meta_path'] = f
    rows.append(r)
d = pd.DataFrame(rows)
d['source_dataset'] = 'figshare_28806086_clinical_gait'
d['source_subject_id'] = d['subject']
d['source_type'] = 'REAL_SOURCE'
d.to_csv(r'metadata\imu_trial_index.csv', index=False)
print('trials:', d.shape[0], '| subjects:', d['subject'].nunique())
g = d.groupby('pathologyKey')
print(g.agg(subjects=('subject', 'nunique'), trials=('trial', 'size'), freq=('freq', lambda s: sorted(s.unique())), score=('evaluationScoreName', lambda s: sorted(s.astype(str).unique())), age_min=('age', 'min'), age_max=('age', 'max')).to_string())
print(d.groupby('pathologyKey')['pathology'].unique().to_string())
print(d.groupby('pathologyKey')['gender'].value_counts().to_string())
