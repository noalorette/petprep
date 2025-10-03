# PETPrep Tutorial

This tutorial will guide you through running **PETPrep** on a real dataset from [OpenNeuro dataset: ds004868](https://openneuro.org/datasets/ds004868/versions/1.0.4).
This dataset is already **BIDS-valid**, the main prerequisite of running PETPrep.

We’ll cover:

1. Installing PETPrep
2. Downloading data from OpenNeuro
3. Running PETPrep commands
4. Inspecting outputs and reports

---

## 1. Install PETPrep

PETPrep is distributed as a **BIDS App container**. You can run it via **Docker** (preferred for local machines) or **Apptainer** (common on HPC systems where Docker is not allowed).

### Install **Docker** / **Apptainer**

* For **Docker**, follow the instructions here: [Docker Desktop installation guide](https://docs.docker.com/get-docker/).
* For **Apptainer** (previously Singularity), follow the instructions here: [Apptainer installation guide](https://apptainer.org/docs/admin/main/installation.html).

To check if Docker or Apptainer is installed, run:

```
docker --version
apptainer --version
```

### Pull the PETPrep image

**Docker:**

```
docker pull ghcr.io/nipreps/petprep:latest
```

**Apptainer:**

```
apptainer pull petprep-latest.sif docker://ghcr.io/nipreps/petprep:latest
```

More details: [PETPrep Installation Docs](https://petprep.readthedocs.io/en/latest/installation.html#containerized-execution-docker-and-apptainer)

---

## 2. Download data from OpenNeuro

We will use dataset **[ds004868 v1.0.4](https://openneuro.org/datasets/ds004868/versions/1.0.4)**.

### Option A: Command line

Install the OpenNeuro client (requires Node.js):

```
npm install -g openneuro-cli
```

Download the dataset:

```
openneuro download --dataset ds004868 --snapshot 1.0.4 /bidsdata
```

For more info: [OpenNeuro CLI Documentation](https://docs.openneuro.org/packages/openneuro-cli.html)

### Option B: Web download

1. Go to the dataset page: [ds004868 v1.0.4](https://openneuro.org/datasets/ds004868/versions/1.0.4)
2. Click **Download dataset**
3. Extract the archive into `/bidsdata`

### Verify your data

```
ls /bidsdata
```

Expected files:

![result download](_static/tutorial_images/BIDS_valid_dataset_from_OpenNeuro.png)

Note that this dataset is not entirely bids valid, as the CHANGES and LICENSE should not be in the root. Move them elsewhere before running PETprep.

---

## 3. Running PETPrep commands

PETPrep follows the standard BIDS App command pattern:

```
<app> <bids_dir> <outputs_dir> <analysis_level> [OPTIONS]
```

### **Docker** example (single subject)

```
docker run -it --rm \
  -v /bidsdata:/data:ro \
  -v /outputs:/out \
  -v /scratch:/work \
  -v /license.txt:/license.txt:ro \
  ghcr.io/nipreps/petprep:latest \
  /data /out participant \
  --participant-label sub-01 \
  --fs-license-file /license.txt \
  --work-dir /work
```

### **Apptainer** example (single subject)

```
apptainer run \
  -B /bidsdata:/data:ro \
  -B /outputs:/out \
  -B /scratch:/work \
  -B /license.txt:/license.txt:ro \
  petprep-latest.sif \
  /data /out participant \
  --participant-label sub-01 \
  --fs-license-file /license.txt \
  --work-dir /work
```

### in this example, we called the whole folder via apptainer as follows:

```
apptainer run \
  --bind /petprep/tutorial_out/ds004868-1.0.4_derivatives:/out \
  --bind /petprep/tutorial_data/ds004868-1.0.4/:/data:ro \
  --bind /petprep/freesurfer_license/license.txt:/license \
  --bind /petprep/tutorial_work/:/work \
/petprep/code/petprep-latest.sif /data /out participant \
  --fs-license-file /license \
  --work-dir /work \
  --ref-mask-name cerebellum \
  --nprocs 20
```

- `--ref-mask-name cerebellum` uses cerebellum as reference tissue for normalization. Choose the reference region depending on the type of tracer used in the dataset.
- `--nprocs 20` runs 20 tasks in parallel on 20 individual physical cores. Note that too high parallel processes can overload system if cores or storage are not available.


## 4. Inspecting outputs and reports

After the run finishes, check the output directory (e.g., `/outputs`):
* The **HTML report** (`report.html`) can be opened in a web browser.
* QC figures (e.g., coregistration, motion plots) are inside the `figures/` folder.
* Preprocessed PET images and transformations are in the `petprep/` folder.


After each run, PETPrep generates an **HTML report**. This file summarizes both anatomical and PET preprocessing, and provides QC figures. Opening the report in your browser is usually the easiest way to check data quality.

### Summary section
The top of the report lists:
- **Subject ID** and sessions processed
- Structural images (e.g. T1-weighted scans)
- Functional/PET series and number of frames
- Output spaces (e.g. native T1w and MNI152 template)
- FreeSurfer reconstruction status

This gives you a quick overview of what PETPrep detected and processed.

---

### Anatomical QC

- **Brain mask and brain tissue segmentation of the T1w**  
  ![Tissue segmentation](_static/tutorial_images/sub-PSTRT01_ses-baseline_dseg.svg)  
  Shows the brain mask and tissue segmentation (GM, WM, CSF). Check that boundaries look plausible.

- **Spatial normalization of the anatomical T1w reference**  
  ![MNI normalization](_static/tutorial_images/sub-PSTRT01_ses-baseline_space-MNI152NLin2009cAsym_T1w.svg)  
  Displays alignment of the T1w image to the MNI template. Check for good overlap. Note: On GitHib this figure looks static, but if you open it locally on your browser, it will toggle between subject and template. 


- **Surface reconstruction**  
  ![Recon-all surfaces](_static/tutorial_images/sub-PSTRT01_ses-baseline_desc-reconall_T1w.svg)  
  Shows white/pial surfaces reconstructed with FreeSurfer on top of the participants T1w template. Misplacements can indicate surface errors.

---

### PET QC

- **Carpet plot**  
  ![PET carpet plot](_static/tutorial_images/sub-PSTRT01_ses-baseline_desc-carpetplot_pet.svg)  
  PETprep implements consecutive frame motion correction, meaning that each frame is aligned to the one before it. This also means that there's no single reference frame for motion correction! Using the flag `--hmc-start-time`, which defaults to 120 sec, you can set the starting point for head motion correction.

How to read the first graph:
    - GS = Global Signal, The average time series across the whole brain mask. Used to check for scanner drifts, global artifacts, or physiological noise.
    - CSF = Cerebrospinal Fluid signal. Average signal extracted from CSF voxels. Often used as a nuisance regressor because CSF is dominated by noise rather than neuronal activity or tracer uptake.
    - WM = White Matter signal. Average signal from white matter voxels. Another typical nuisance regressor; helps identify noise contributions unrelated to gray matter metabolism or binding.
    - DVARS = D temporal VARiance of Signal. Measures how much the signal intensity changes frame-to-frame across the whole brain. Spikes indicate possible motion or sudden scanner artifacts.
    - FD = Framewise Displacement. A head-motion metric summarizing how much the subject’s head moved between frames (translations + rotations combined). High peaks → strong motion events.


How to read the second graph:
If the carpet plot shows smooth shading transitions across frames, that indicates motion was minimal and tracer uptake is stable. In contrast, sudden vertical bands or shifts are a clear sign of motion artifacts. Uneven brightness across different tissue classes reflects tracer-specific uptake dynamics.
    - Ctx GM (blue) = cortical gray matter
    - dGM (orange) = deep gray matter (e.g., thalamus, basal ganglia)
    - sWM + sCSF (green) = surrounding white matter + subcortical CSF
    - Cb (red) = cerebellum
    - Edge (purple) = voxels at the brain boundary (where partial volume or motion artifacts are more common)

- **Confound correlation**  
  ![Confound correlation](_static/tutorial_images/sub-PSTRT01_ses-baseline_desc-confoundcorr_pet.svg)  


Motion confounds:
  These include translations (x, y, z), rotations (x, y, z), and composite metrics like FD (framewise displacement) or RMSD. They capture head movements: the subject shifts, nods, or rotates. If motion regressors (FD, rotations, translations) had been highly correlated with global signal, that would indicate global signal is contaminated by head motion.

Physiological confounds:
  These include global signal, CSF, and white matter time series. They capture fluctuations caused by respiration and cardiac pulsation (blood flow changes), scanner drift, systemic metabolism changes (especially important in PET where uptake evolves with time).

By ranking correlations with global signal in graph 2, you can see whether motion or physiology is the main driver of motion. Weak motion–global signal correlation suggests motion correction worked well! Note that shading in the right plot lets you distinguish whether a confound increases with global signal (positive correlation) or moves in the opposite direction (negative correlation), the bar heights all represent absolute magnitudes.


- **PET to MRI co-registration**  
  ![Coregistration](_static/tutorial_images/sub-PSTRT01_ses-baseline_desc-coreg_pet.svg)  
  Shows alignment between the PET reference and T1w MRI. Note that this image is not static when opened in a browser.


- **Reference mask check**  
  ![Reference mask](_static/tutorial_images/sub-PSTRT01_ses-baseline_ref-cerebellum_desc-refmask_pet.svg)  
  Reference region mask overlaid on the PET reference and anatomical. Note that this image is not static when opened in a browser.


---  

