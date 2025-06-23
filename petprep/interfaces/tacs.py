import pandas as pd
import nibabel as nb
import numpy as np
import json
from nipype.interfaces.base import BaseInterfaceInputSpec, File, SimpleInterface, TraitedSpec
from nipype.utils.filemanip import fname_presuffix


class _ExtractTACsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='PET file in anatomical space')
    segmentation = File(exists=True, mandatory=True, desc='Segmentation in anatomical space')
    dseg_tsv = File(exists=True, mandatory=True, desc='Lookup table for segmentation')
    metadata = File(exists=True, mandatory=True, desc='PET JSON metadata file')


class _ExtractTACsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Regional time activity curves')


class ExtractTACs(SimpleInterface):
    """Extract time activity curves from a segmentation."""

    input_spec = _ExtractTACsInputSpec
    output_spec = _ExtractTACsOutputSpec

    def _run_interface(self, runtime):
        pet_img = nb.load(self.inputs.in_file)
        pet_data = pet_img.get_fdata(dtype='float32')
        if pet_data.ndim == 3:
            pet_data = pet_data[..., np.newaxis]
        seg = nb.load(self.inputs.segmentation).get_fdata().astype('int32')

        seginfo = pd.read_csv(self.inputs.dseg_tsv, sep='\t')
        indices = seginfo.iloc[:, 0].to_numpy()
        names = seginfo.iloc[:, 1].tolist()

        with open(self.inputs.metadata) as f:
            metadata = json.load(f)
        frame_times = metadata.get('FrameTimesStart', [])
        frame_durations = metadata.get('FrameDuration', [])

        flat_pet = pet_data.reshape(-1, pet_data.shape[3])
        flat_seg = seg.reshape(-1)
        curves = {}
        for idx, name in zip(indices, names):
            mask = flat_seg == idx
            if np.any(mask):
                curves[name] = flat_pet[mask].mean(axis=0)
            else:
                curves[name] = np.full(pet_data.shape[3], np.nan, dtype=float)

        frame_times_end = [s + d for s, d in zip(frame_times, frame_durations)]
        df = pd.DataFrame(curves)
        df.insert(0, 'FrameTimesEnd', frame_times_end)
        df.insert(0, 'FrameTimesStart', list(frame_times))

        out_file = fname_presuffix(self.inputs.in_file, suffix='_timeseries.tsv', newpath=runtime.cwd, use_ext=False)
        df.to_csv(out_file, sep='\t', index=False, na_rep='n/a')

        self._results['out_file'] = out_file
        return runtime


__all__ = ('ExtractTACs',)