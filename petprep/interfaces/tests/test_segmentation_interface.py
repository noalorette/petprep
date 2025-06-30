import pytest
from ..segmentation import SegmentGTM
from pathlib import Path

def test_segmentgtm_skip(tmp_path):
    subj_dir = tmp_path / "sub-01"
    (subj_dir / "mri").mkdir(parents=True)
    (subj_dir / "stats").mkdir()
    (subj_dir / "mri" / "gtmseg.mgz").write_text("")
    (subj_dir / "stats" / "gtmseg.stats").write_text("")

    seg = SegmentGTM(subjects_dir=str(tmp_path), subject_id="sub-01")
    res = seg.run()

    assert res.runtime.returncode == 0
    assert Path(res.outputs.out_file) == subj_dir / "mri" / "gtmseg.mgz"