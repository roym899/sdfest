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

# Prepare Datasets
See below for expected folder structure for each dataset.

### ShapeNet ([Website](https://shapenet.org/))
```
./data/shapenet/bottle_02876657/...
./data/shapenet/bowl_02880940/...
./data/shapenet/mug_03797390/...
```

### ModelNet ([Website](https://modelnet.cs.princeton.edu/))
```
./data/modelnet/bottle/...
./data/modelnet/bowl/...
./data/modelnet/cup/...
```

### Redwood ([Website](http://redwood-data.org/3dscan/dataset.html))
```
./data/bottle/rgbd/...
./data/bowl/rgbd/...
./data/mug/rgbd/...
```

### RGB-D Object Dataset ([Website](https://rgbd-dataset.cs.washington.edu/index.html))
```
./data/rgbd_objects_uw/{bowl,coffee_mug,shampoo,water_bottle}/{bowl,coffee_mug,shampoo,water_bottle}_1/
./data/rgbd_objects_uw/{bowl,coffee_mug,shampoo,water_bottle}/{bowl,cofee_mug,shampoo,water_bottle}_2/
...
```

# Reproduce Experiments
Depending on which dataset, you have downloaded you can reproduce the results reported in the paper (using the already trained models) by running the script
```bash
source reproduce_{shapenet,modelnet,redwood,rgbd_uw}_experiments.sh
```
after that, all results can be found in `./results`.

# Train models
To train a network for a specific category you need to first train a per-category VAE, and *afterwards* an initialization network.
## VAE

## Init

# Code Structure
Code is structured into 5 standalone Python packages:
- *sdf_vae*: variational auto-encoder for shapes
- *sdf_differentiable_renderer*: differentiable renderer for discretized SDF
- *sdf_single_shot*: pose and shape estimation from point cloud
- *sdf_estimation*: integration of VAE, differentiable renderer and single shot network into initialization + render and compare pipeline
- *yoco*: simple configuration tool