"""Neuron 3D positions (micrometers) in the same order as the brain model -> build/pos.npz"""
import numpy as np, pandas as pd, os
os.makedirs('build', exist_ok=True)
comp = pd.read_csv('raw/Drosophila_brain_model/Completeness_783.csv', index_col=0)
ann = pd.read_csv('raw/flywire_annotations/supplemental_files/Supplemental_file1_neuron_annotations.tsv', sep='\t', low_memory=False)
m = ann.set_index('root_id').reindex(comp.index)
x = m.pos_x.values * 4e-3; y = m.pos_y.values * 4e-3; z = m.pos_z.values * 40e-3   # FlyWire voxels are 4x4x40 nm
ok = ~np.isnan(x)
np.savez('build/pos.npz', x=x, y=y, z=z, ok=ok, super_class=m.super_class.fillna('').values.astype(str))
print('neurons with positions:', ok.sum())
