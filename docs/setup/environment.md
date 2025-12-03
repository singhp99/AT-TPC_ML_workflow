## Virtual Environment Setup: Spyral

The first step of the ML workflow involves extracting training features and labels, we do so from a package written by many individuals in the AT-TPC group called [Spyral](https://attpc.github.io/Spyral/), which has its own documentation page. 

However, for our purposes, the steps outlined below will be in a condensed form. 

First navigate to the folder on your local or High Performance Cluster (recommended) you would want to be in and clone this repository with

```bash
git clone <repository_url>
```

then 

```bash
cd AT-TPC_ML_workflow
```

We will now be installing the AT-TPC package but do not want to do so globally, instead we will first create a virtual environment.

This package only works for a Python version with in the bounds of > 3.10 and < 3.13 (either 3.11 or 3.12), therefore we must ensure the environment is created with the right version.  

You can check you current version by 

```bash
python --version
```
And then run this command to create the virtual environment

```bash
python3.12 -m venv .venv
```
Or if you are doing this on an HPC, load the correct python version before you activate this environment. This process could be specific to the HPC you use, and you may need to refer to their specific documentation. As an example, for my specific cluster I use
```bash
module load Python/3.12.3-GCCcore-13.3.0
```
then run
```bash
python -m venv .venv
```

Note that it is important you run this command inside the AT-TPC_ML_workflow folder. Now we will activate this environment by 

```bash
source .venv/bin/activate
```

and finally install attpc_spyral and its dependencies with a `requirements.txt`, a file that should be part of the GitHub clone. If you inspect this file, all the dependencies can be seen. 

```bash
pip install -r requirements.txt
```

Congratulations! This installs your virtual environment for the training features and labels extraction. You can close this environment with 

```bash
deactivate
```

Please do this before proceeding to the next step.

## Conda Environment Setup: ML

Unlike the virtual environment for ML training extraction, we will use a Conda environment for the ML side of things, this includes the preprocessing for ML data. 

This environment requires more rigid dependencies' versions than the virtual environment we installed in the step above, hence why we will be using a file named environment.yml to install all the dependencies with their receptive versions. 

Simply run
```bash
conda env create -f environment.yml
```

If you inspect this file, it has all the right dependencies you would need for running ML side of this project, and will create a Conda environment called "tf_Jul2025".

You can exit this environment by similar command to the virtual environment 

```bash
conda deactivate
```