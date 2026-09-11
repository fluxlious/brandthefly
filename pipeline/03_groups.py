"""Named neuron groups used for readouts and presets -> build/groups.json"""
import numpy as np, pandas as pd, json
comp = pd.read_csv('raw/Drosophila_brain_model/Completeness_783.csv', index_col=0)
ann = pd.read_csv('raw/flywire_annotations/supplemental_files/Supplemental_file1_neuron_annotations.tsv', sep='\t', low_memory=False)
m = ann.set_index('root_id').reindex(comp.index).reset_index()
ids = comp.index.values; id2i = {v: k for k, v in enumerate(ids)}
sel = lambda mask: [int(x) for x in np.where(mask)[0]]
ct = m.cell_type.fillna(''); side = m.side.fillna(''); sub = m.cell_sub_class.fillna(''); sc = m.super_class.fillna('')
G = {}
for name, typ in [('DNa01', 'DNa01'), ('DNa02', 'DNa02'), ('MDN', 'MDN'), ('GF', 'DNp01'), ('DNp09', 'DNp09')]:   # steering, backward, giant fiber (escape), forward
    for s in ['left', 'right']:
        G[f'{name}_{s[0].upper()}'] = sel((ct == typ) & (side == s))
G['MN9'] = [id2i[720575940660219265]]                          # proboscis extension motor neuron
G['ingest_MN'] = sel(sub == 'ingestion_motor_neuron'); G['proboscis_MN'] = sel(sub == 'proboscis_motor_neuron')
for s in ['left', 'right']:
    G[f'antenna_MN_{s[0].upper()}'] = sel((sub == 'antennal_motor_neuron') & (side == s))
    G[f'neck_MN_{s[0].upper()}'] = sel((sub == 'neck_motor_neuron') & (side == s))
G['DN_all'] = sel(sc == 'descending')
sugar = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768, 720575940630797113, 720575940632889389,
         720575940621754367, 720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446,
         720575940617937543, 720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663, 720575940611875570]
G['sugar_GRN'] = [id2i[x] for x in sugar if x in id2i]            # sugar-sensing neurons from Shiu et al. example
G['loom'] = sel(ct.isin(['LC4', 'LPLC2']))                         # looming detectors
G['PAM'] = sel(ct.str.startswith('PAM'))                           # reward dopamine neurons
G['PPL1'] = sel(ct.str.startswith('PPL1'))                         # punishment dopamine neurons
json.dump(G, open('build/groups.json', 'w'))
print({k: len(v) for k, v in G.items()})
