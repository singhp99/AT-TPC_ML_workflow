### Virtual Environment Setup: Spyral

The first step of the ML workflow involves extracting training features and labels, we do so from a package written by many individuals in the AT-TPC group called [Spyral](https://attpc.github.io/Spyral/), which has its own documentation page. 

However, for our purposes, the steps outlined below will be in a condensed form. 

First navigate to the folder on your local or High Performance Cluster (recommended) you would want to be in and clone this repository with

```
git clone <repository_url>
```

then 

```
cd AT-TPC_ML_workflow
```

We will now be installing the AT-TPC package but do not want to do so globally, instead we will first create a virtual environment.

```
python -m venv .venv
```
or 

```
python3 -m venv .venv
```

Note that it is important you run this command inside the AT-TPC_ML_workflow folder. Now we will activate this environment by 

```
source .venv/bin/activate
```

and finally install attpc_spyral and its dependencies with

```
pip install attpc_spyral
```
or 
```
pip3 install attpc_spyral
```

Congratulations! This installs your virtual environment for the training features and labels extraction. You can close this environment with 

```
deactivate
```

Please do this before proceeding to the next step.

### Conda Environment Setup: ML

Unlike the virtual environment for ML training extraction, we will use a Conda environment for the ML side of things, this includes the preprocessing for ML data. 


