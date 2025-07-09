from __future__ import annotations

from pathlib import Path

from nilearn.image import resample_to_img
from nipype.interfaces import utility as niu
from nipype.interfaces.utility import Function
from nipype.pipeline import engine as pe

from ...interfaces import ExtractTACs


def resample_pet_to_segmentation(pet_file: str, segmentation_file: str) -> str:
    """Resample the PET image to the segmentation space."""

    resampled_pet = resample_to_img(
        pet_file,
        segmentation_file,
        interpolation='continuous',
    )
    out_file = Path('pet_resampled.nii.gz').absolute()
    resampled_pet.to_filename(out_file)
    return str(out_file)


def init_pet_tacs_wf(*, name: str = 'pet_tacs_wf') -> pe.Workflow:
    """Create a workflow to extract time–activity curves from a segmentation.

    Parameters
    ----------
    name : :obj:`str`
        Name of workflow (default: ``pet_tacs_wf``)

    Returns
    -------
    workflow : :class:`~nipype.pipeline.engine.Workflow`
        The TAC extraction workflow.
    """

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['pet_anat', 'segmentation', 'dseg_tsv', 'metadata']),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['timeseries']), name='outputnode')

    # Resample PET to segmentation space
    resample_pet = pe.Node(
        Function(
            input_names=['pet_file', 'segmentation_file'],
            output_names=['resampled_pet'],
            function=resample_pet_to_segmentation,
        ),
        name='resample_pet',
    )

    tac = pe.Node(
        ExtractTACs(),
        name='tac',
    )

    workflow.connect(
        [
            (
                inputnode,
                resample_pet,
                [('pet_anat', 'pet_file'), ('segmentation', 'segmentation_file')],
            ),
            (
                resample_pet,
                tac,
                [('resampled_pet', 'in_file')],
            ),
            (
                inputnode,
                tac,
                [
                    ('segmentation', 'segmentation'),
                    ('dseg_tsv', 'dseg_tsv'),
                    ('metadata', 'metadata'),
                ],
            ),
            (tac, outputnode, [('out_file', 'timeseries')]),
        ]
    )

    return workflow


__all__ = ('init_pet_tacs_wf',)
