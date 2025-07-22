from nipype.interfaces.base import File, traits, isdefined
from nireports.interfaces.reporting.base import (
    SimpleBeforeAfterRPT,
    _SimpleBeforeAfterInputSpecRPT,
)
from nireports.interfaces.reporting.masks import (
    SimpleShowMaskRPT,
    _SimpleShowMaskInputSpec,
)


class _BeforeAfterMaskInputSpec(_SimpleBeforeAfterInputSpecRPT):
    roi_mask = File(desc="mask defining cropping region")


class BeforeAfterMaskRPT(SimpleBeforeAfterRPT):
    input_spec = _BeforeAfterMaskInputSpec

    def _post_run_hook(self, runtime):
        if isdefined(self.inputs.roi_mask):
            self._fixed_image_mask = self.inputs.roi_mask
        return super()._post_run_hook(runtime)


class _ShowMaskRPTROIInputSpec(_SimpleShowMaskInputSpec):
    roi_mask = File(desc="mask defining cropping region")


class ShowMaskRPTROI(SimpleShowMaskRPT):
    input_spec = _ShowMaskRPTROIInputSpec

    def _post_run_hook(self, runtime):
        self._anat_file = self.inputs.background_file
        self._seg_files = [self.inputs.mask_file]
        self._mask_file = (
            self.inputs.roi_mask
            if isdefined(self.inputs.roi_mask)
            else self.inputs.mask_file
        )
        self._masked = True
        return super(SimpleShowMaskRPT, self)._post_run_hook(runtime)
