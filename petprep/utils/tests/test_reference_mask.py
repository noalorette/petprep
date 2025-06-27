import numpy as np
import nibabel as nib

from petprep.utils.reference_mask import generate_reference_region


def test_generate_reference_region_target_volume():
    seg = np.zeros((10, 10, 10), dtype=np.int16)
    seg[3:7, 3:7, 3:7] = 1
    img = nib.Nifti1Image(seg, np.eye(4))

    config = {
        "refmask_indices": [1],
        "smooth_fwhm_mm": 2.3548,
        "target_volume_ml": 0.04,
    }

    out_img = generate_reference_region(img, config)
    mask = out_img.get_fdata()
    voxel_vol_ml = np.prod(out_img.header.get_zooms()) / 1000.0
    volume_ml = mask.sum() * voxel_vol_ml
    assert np.isclose(volume_ml, config["target_volume_ml"], atol=0.001)