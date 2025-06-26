from petprep.interfaces.refmask import DilateMask, ErodeMask


def load_pvc_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)


def init_pet_pvc_wf(
    *,
    segmentation: str = 'gtm',
    config,
    name: str = 'pet_refmask_wf',
) -> pe.Workflow:
    config = load_refmask_config(config_path)


    inputnode = pe.Node(
        niu.IdentityInterface(fields=['segmentation', 'refmask_index', ]),
        name='inputnode'
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['out_refmask']),
        name='outputnode'
    )



    return workflow