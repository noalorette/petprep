import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import binary_dilation, binary_erosion, ball


def _clip(mask: np.ndarray, between=(0, 1)) -> np.ndarray:
    return np.clip(mask, *between)


def generate_reference_region(
    seg_img: nib.Nifti1Image,
    config: dict
) -> nib.Nifti1Image:
    """Generate a reference region using a flexible config.

    Config keys:
        - refmask_indices (list[int]): Labels to include in the reference region.
        - exclude_indices (list[int], optional): Labels to subtract (after dilation).
        - erode_by_voxels (int, optional): Number of voxels to erode the target region.
        - dilate_by_voxels (int, optional): Dilation radius for excluded regions.
        - smooth_fwhm_mm (float, optional): FWHM for smoothing the target region.
        - target_volume_ml (float, optional): Keep only the top N voxels by smoothed value.

    Returns:
        nib.Nifti1Image: Final reference mask.
    """

    data = seg_img.get_fdata()
    affine = seg_img.affine
    header = seg_img.header
    zooms = header.get_zooms()
    voxel_vol_mm3 = np.prod(zooms)

    # Step 1: Create binary target mask
    mask = np.isin(data, config["refmask_indices"]).astype(np.uint8)

    # Step 2: Optional erosion
    if "erode_by_voxels" in config and config["erode_by_voxels"] > 0:
        mask = binary_erosion(mask, ball(config["erode_by_voxels"])).astype(np.uint8)

    # Step 3: Optional exclusion
    if "exclude_indices" in config and config["exclude_indices"]:
        exclude = np.isin(data, config["exclude_indices"]).astype(np.uint8)
        if "dilate_by_voxels" in config and config["dilate_by_voxels"] > 0:
            exclude = binary_dilation(exclude, ball(config["dilate_by_voxels"])).astype(np.uint8)
        mask = mask - exclude
        mask = _clip(mask)

    # Step 4: Optional smoothing + volume constraint
    if "smooth_fwhm_mm" in config and "target_volume_ml" in config:
        sigma = config["smooth_fwhm_mm"] / (2.3548 * zooms[0])  # approximate sigma from FWHM
        smoothed = gaussian_filter(mask.astype(np.float32), sigma=sigma)

        target_voxels = int((config["target_volume_ml"] * 1000) / voxel_vol_mm3)
        threshold = np.sort(smoothed[mask > 0].flatten())[-target_voxels]
        mask = ((smoothed >= threshold) & (mask > 0)).astype(np.uint8)

    return nib.Nifti1Image(mask, affine, header)
