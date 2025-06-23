from __future__ import annotations

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from ... import config
from ...interfaces import DerivativesDataSink, ExtractTACs
from .outputs import prepare_timing_parameters


def init_pet_tacs_wf(*, output_dir: str, metadata: dict, name: str = 'pet_tacs_wf') -> pe.Workflow:
    """Extract time activity curves from a segmentation."""

    timing_parameters = prepare_timing_parameters(metadata)

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['source_file', 'pet_anat', 'segmentation', 'dseg_tsv']
        ),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['timeseries']), name='outputnode')

    tac = pe.Node(
        ExtractTACs(
            frame_times=timing_parameters.get('FrameTimesStart', []),
            frame_durations=timing_parameters.get('FrameDuration', []),
        ),
        name='tac',
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )

    ds_tac = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc='timeactivity',
            suffix='timeseries',
        ),
        name='ds_tac',
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )

    workflow.connect([
        (inputnode, tac, [
            ('pet_anat', 'in_file'),
            ('segmentation', 'segmentation'),
            ('dseg_tsv', 'dseg_tsv'),
        ]),
        (inputnode, ds_tac, [('source_file', 'source_file')]),
        (tac, ds_tac, [('out_file', 'in_file')]),
        (tac, outputnode, [('out_file', 'timeseries')]),
    ])

    return workflow


__all__ = ('init_pet_tacs_wf',)
