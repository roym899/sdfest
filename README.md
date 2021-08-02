# SDFEst
Shape and pose estimation using discretized signed distance fields.

# Installation

## Using Poetry
Easiest way to install a tested setup and reproduce the experiments is to use Poetry.

First, [install Poetry](https://python-poetry.org/docs/#installation) and execute the following command inside the root directory.
```bash
poetry install
```

## Using pip
You can also install everything with pip (either in your global Python environment, or a virtual environment).
This will get the latest version of all dependencies though, so if a breaking change was introduced this might stop working.

Execute the following command to install the packages and their dependencies.
```bash
pip install -e ./yoco/
pip install -e ./sdf_differentiable_renderer/
pip install -e ./sdf_vae/
pip install -e ./sdf_single_shot/
pip install -e ./sdf_estimation/
```

## Optional: Detectron2
You need to install Detectron2 manually inside the Poetry environment to run the whole pipeline on real data. 

If you use Poetry, first use
```bash
poetry shell
```
or activate your virtual environment.

Then follow the [detectron2 installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) from there.
Tested with detectron2 0.5 + torch 1.9.0.
