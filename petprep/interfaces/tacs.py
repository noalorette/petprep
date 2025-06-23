from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from niworkflows.utils.timeseries import _nifti_timeseries


class _ExtractTACsInputSpec(BaseInterfaceInputSpec):
    in_pet = File(exists=True, mandatory=True, desc='input 4D PET image')
    segmentation = File(exists=True, mandatory=True, desc='segmentation defining regions')
    out_file = File(desc='output TSV file')


class _ExtractTACsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output TSV file with extracted TACs')


class ExtractTACs(SimpleInterface):
    """Extract time-activity curves (TACs) from a PET series."""

    input_spec = _ExtractTACsInputSpec
    output_spec = _ExtractTACsOutputSpec

    def _run_interface(self, runtime):
        pet_img = nb.load(self.inputs.in_pet)
        seg_img = nb.load(self.inputs.segmentation)

        data, segments = _nifti_timeseries(pet_img, seg_img)

        tacs = {
            label: np.mean(data[idx], axis=0)
            for label, idx in segments.items()
        }
        df = pd.DataFrame(tacs)

        out_file = self.inputs.out_file
        if not out_file:
            out_file = Path(runtime.cwd) / 'tacs.tsv'
        else:
            out_file = Path(runtime.cwd) / out_file

        df.to_csv(out_file, sep='\t', index=False)
        self._results['out_file'] = str(out_file)
        return runtime


__all__ = ('ExtractTACs',)