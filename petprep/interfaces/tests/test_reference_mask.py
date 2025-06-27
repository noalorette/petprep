import json
from pathlib import Path

import nibabel as nb
import numpy as np
from nipype.pipeline import engine as pe

from ..reference_mask import ExtractRefRegion


def _create_seg(tmp_path: Path) -> Path:
    data = np.zeros((5, 5, 5), dtype=int)
    data[1, 1, 1] = 1
    data[2, 2, 2] = 2
    img = nb.Nifti1Image(data, np.eye(4))
    seg_file = tmp_path / "seg.nii.gz"
    img.to_filename(seg_file)
    return seg_file


def _create_config(tmp_path: Path, indices):
    cfg = {"testseg": {"region": {"refmask_indices": indices}}}
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(cfg))
    return cfg_file


def test_extract_refregion(tmp_path):
    seg = _create_seg(tmp_path)
    cfg = _create_config(tmp_path, [1])

    node = pe.Node(
        ExtractRefRegion(
            seg_file=str(seg),
            config_file=str(cfg),
            segmentation_type="testseg",
            region_name="region",
        ),
        base_dir=str(tmp_path),
    )
    res = node.run()
    out = nb.load(res.outputs.refmask_file).get_fdata()
    assert out[1, 1, 1] == 1
    assert out.sum() == 1


def test_extract_refregion_override(tmp_path):
    seg = _create_seg(tmp_path)
    cfg = _create_config(tmp_path, [1])

    node = pe.Node(
        ExtractRefRegion(
            seg_file=str(seg),
            config_file=str(cfg),
            segmentation_type="testseg",
            region_name="region",
            override_indices=[2],
        ),
        base_dir=str(tmp_path),
    )
    res = node.run()
    out = nb.load(res.outputs.refmask_file).get_fdata()
    assert out[2, 2, 2] == 1
    assert out.sum() == 1