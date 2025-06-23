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
            fields=['pet_anat', 'segmentation', 'dseg_tsv', 'metadata']
        ),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['timeseries']), name='outputnode')

    tac = pe.Node(
        ExtractTACs(),
        name='tac',
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )

    workflow.connect(
            [
                (
                    inputnode,
                    tac,
                    [
                        ('pet_anat', 'in_file'),
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
