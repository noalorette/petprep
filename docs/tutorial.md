# PETPrep Tutorial 

This tutorial will guide you through running **PETPrep** on a real dataset from [OpenNeuro dataset: ds005605`](https://openneuro.org/datasets/ds005605/versions/1.1.0). This dataset is already **BIDS-valid**, the main prerequisite of running PETPrep.


We’ll cover:
1. Installing PETPrep
2. Downloading data from OpenNeuro
3. Understanding folder structures
4. Running PETPrep commands
5. Inspecting outputs and reports

---

## 1. Install PETPrep

PETPrep is distributed as a **BIDS App container**. You can run it via **Docker** (preferred) or **Apptainer** (if Docker can't be installed on HPC). 

### Install Docker/Apptainer

 - For Docker, follow the instructions here: [Docker Desktop installation guide](https://docs.docker.com/get-docker/).  
 - For Apptainer (previously Singularity), follow the instructions here: [Apptainer installation guide](https://apptainer.org/docs/admin/main/installation.html).

To check if Apptainer or Docker is installed, run:

```bash
docker --version
apptainer --version
```
### Then pull the image 

```bash
docker pull ghcr.io/nipreps/petprep:latest
```
You can find more documentation pulling the image via Docker or Apptainer [here](https://petprep.readthedocs.io/en/latest/installation.html#containerized-execution-docker-and-apptainer).

