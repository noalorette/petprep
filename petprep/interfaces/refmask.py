import os
import numpy as np
import nibabel as nib
from nipype.interfaces.base import (
    SimpleInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    traits,
    isdefined,
)


def _clip(mask: np.ndarray, between: tuple = (0, 1)) -> np.ndarray:
    return np.clip(mask, *between)


# Dilation Interface
from skimage.morphology import binary_dilation, ball


class DilateMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="Input 3D binary mask (NIfTI)")
    by_radius = traits.Int(mandatory=True, desc="Radius for dilation")


class DilateMaskOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Dilated mask")


class DilateMask(SimpleInterface):
    input_spec = DilateMaskInputSpec
    output_spec = DilateMaskOutputSpec

    def _run_interface(self, runtime):
        img = nib.load(self.inputs.in_file)
        data = img.get_fdata()
        dilated = binary_dilation(data, ball(self.inputs.by_radius))
        dilated = _clip(dilated).astype(np.uint8)

        out_path = os.path.abspath("dilated_mask.nii.gz")
        nib.save(nib.Nifti1Image(dilated, img.affine, img.header), out_path)
        self._results["out_file"] = out_path
        return runtime


# Erosion Interface
from skimage.morphology import binary_erosion


class ErodeMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="Input 3D binary mask (NIfTI)")
    by_radius = traits.Int(mandatory=True, desc="Radius for erosion")


class ErodeMaskOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Eroded mask")


class ErodeMask(SimpleInterface):
    input_spec = ErodeMaskInputSpec
    output_spec = ErodeMaskOutputSpec

    def _run_interface(self, runtime):
        img = nib.load(self.inputs.in_file)
        data = img.get_fdata()
        eroded = binary_erosion(data, ball(self.inputs.by_radius))
        eroded = _clip(eroded).astype(np.uint8)

        out_path = os.path.abspath("eroded_mask.nii.gz")
        nib.save(nib.Nifti1Image(eroded, img.affine, img.header), out_path)
        self._results["out_file"] = out_path
        return runtime