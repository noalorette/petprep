from nipype.interfaces.base import (
    SimpleInterface, BaseInterfaceInputSpec, TraitedSpec,
    File, traits
)
import os
import nibabel as nib
import numpy as np
import json


class ExtractRefRegionInputSpec(BaseInterfaceInputSpec):
    seg_file = File(exists=True, mandatory=True, desc="Segmentation NIfTI file")
    config_file = File(exists=True, mandatory=True, desc="Path to the config.json file")
    segmentation_type = traits.Str(mandatory=True, desc="Type of segmentation (e.g. 'gtm', 'wm')")
    region_name = traits.Str(mandatory=True, desc="Name of the reference region (e.g. 'cerebellum')")


class ExtractRefRegionOutputSpec(TraitedSpec):
    refmask_file = File(exists=True, desc="Output reference mask NIfTI file")


class ExtractRefRegion(SimpleInterface):
    input_spec = ExtractRefRegionInputSpec
    output_spec = ExtractRefRegionOutputSpec

    def _run_interface(self, runtime):
        seg_img = nib.load(self.inputs.seg_file)

        # Load the config
        with open(self.inputs.config_file, "r") as f:
            config = json.load(f)

        try:
            cfg = config[self.inputs.segmentation_type][self.inputs.region_name]
        except KeyError:
            raise ValueError(f"Configuration not found for segmentation='{self.inputs.segmentation_type}' "
                             f"and region='{self.inputs.region_name}'")

        # Extract configuration parameters
        target_labels = cfg.get("refmask_indices", [])
        surrounding_labels = cfg.get("exclude_indices", [])
        erode = cfg.get("erode_by_voxels", 0)
        dilate = cfg.get("dilate_by_voxels", 0)

        from .your_module import extract_refregion_from_segmentation  # adjust this import

        refmask_img = extract_refregion_from_segmentation(
            seg_img=seg_img,
            target_labels=target_labels,
            surrounding_labels=surrounding_labels,
            erode_by_voxels=erode,
            dilate_surrounding_by_voxels=dilate
        )

        out_file = os.path.abspath("refmask.nii.gz")
        nib.save(refmask_img, out_file)
        self._results["refmask_file"] = out_file
        return runtime
