import json
import pandas as pd
import numpy as np
import nibabel as nb
from nipype.pipeline import engine as pe

from petprep.interfaces.tacs import ExtractTACs


def test_ExtractTACs(tmp_path):
    pet_data = np.stack([
        np.ones((2, 2, 2)),
        np.ones((2, 2, 2)) * 2,
    ], axis=-1)
    pet_file = tmp_path / 'pet.nii.gz'
    nb.Nifti1Image(pet_data, np.eye(4)).to_filename(pet_file)

    seg_data = np.tile([[1, 2], [1, 2]], (2, 1, 1)).astype('int16')
    seg_file = tmp_path / 'seg.nii.gz'
    nb.Nifti1Image(seg_data, np.eye(4)).to_filename(seg_file)

    dseg_tsv = tmp_path / 'seg.tsv'
    pd.DataFrame({'index': [1, 2], 'name': ['A', 'B']}).to_csv(dseg_tsv, sep='\t', index=False)

    meta_json = tmp_path / 'pet.json'
    meta_json.write_text(json.dumps({'FrameTimesStart': [0, 1], 'FrameDuration': [1, 1]}))

    node = pe.Node(
        ExtractTACs(
            in_file=str(pet_file),
            segmentation=str(seg_file),
            dseg_tsv=str(dseg_tsv),
            metadata=str(meta_json),
        ),
        name='tac',
        base_dir=tmp_path,
    )
    res = node.run()

    out = pd.read_csv(res.outputs.out_file, sep='\t')
    assert list(out.columns) == ['FrameTimesStart', 'FrameTimesEnd', 'A', 'B']
    assert np.allclose(out['A'], [1, 2])
    assert np.allclose(out['B'], [1, 2])