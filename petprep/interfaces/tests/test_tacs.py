import numpy as np
import pandas as pd
import nibabel as nb
from nipype.pipeline import engine as pe

from petprep.interfaces.tacs import ExtractTACs


def test_ExtractTACs(tmp_path):
    pet_file = tmp_path / 'pet.nii.gz'
    seg_file = tmp_path / 'seg.nii.gz'

    data = np.arange(24, dtype='float32').reshape((2, 3, 4, 1))
    nb.Nifti1Image(data, np.eye(4)).to_filename(pet_file)

    seg = np.zeros((2, 3, 4), dtype='int16')
    seg[0, 0, 0] = 1
    seg[0, 0, 1:] = 2
    nb.Nifti1Image(seg, np.eye(4)).to_filename(seg_file)

    node = pe.Node(
        ExtractTACs(in_pet=str(pet_file), segmentation=str(seg_file)),
        name='extract',
        base_dir=tmp_path,
    )
    res = node.run()

    df = pd.read_csv(res.outputs.out_file, sep='\t')
    assert 'Ctx GM' in df.columns and 'dGM' in df.columns
    assert np.allclose(df['Ctx GM'].to_numpy(), 0.0)
    assert np.allclose(df['dGM'].to_numpy(), 2.0)