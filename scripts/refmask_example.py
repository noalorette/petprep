#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 15:52:39 2025

@author: martinnorgaard
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 09:45:55 2025

@author: martinnorgaard
"""

import os
from nipype import Workflow, Node
from nipype.interfaces.utility import IdentityInterface

def main():
    # Define paths to your input files
    segmentation = 'sub-010_ses-baseline_desc-gtm_dseg.nii.gz'
    
    output_dir = 'test_data/'
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the workflow
    workflow = init_refmask_wf(output_dir=output_dir, metadata={}, name='test_refmask_wf')

    # Set inputs to the workflow
    workflow.inputs.inputnode.segmentation = segmentation


    # Run the workflow
    workflow.base_dir = os.path.join(output_dir, 'work')
    workflow.run()

    print("Workflow execution completed.")

if __name__ == '__main__':
    main()