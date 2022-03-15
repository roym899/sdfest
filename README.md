# SDFEst
Shape and pose estimation using discretized signed distance fields.

## Installation

You can also install everything with pip (either in your global Python environment, or a virtual environment).

### Non-editable
```
pip install sdfest
```
In principle, this allows to train novel networks, apply the pipeline, use the individual modules, and reproduce our experiments.

### Editable (i.e., for modifying / experimenting with the code):
```
git clone git@github.com:roym899/sdfest.git
cd sdfest
pip install -e .
```
This will get the latest version of all dependencies, which might break if a breaking change was introduced in a dependency.
To reproduce our tested environment you can use the provided requirements.txt.

Execute the following command to install the packages and their dependencies. Note that this will likely downgrade / upgrade versions in your current environment, so it's better to use this in a virtual environment only.
```bash
git clone git@github.com:roym899/sdfest.git
cd sdfest
pip install -r requirements.txt -e .
```

### Optional: Detectron2
You need to install Detectron2 manually to run the pipeline with automatic instance segmentation.

Follow the [detectron2 installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) from there.
Tested with detectron2 0.5 + torch 1.9.0.

## Prepare Datasets
See below for expected folder structure for each dataset.

#### ShapeNet ([Website](https://shapenet.org/))
```
./data/shapenet/02876657/...
./data/shapenet/02880940/...
./data/shapenet/03797390/...
```

#### ModelNet ([Website](https://modelnet.cs.princeton.edu/))
```
./data/modelnet/bottle/...
./data/modelnet/bowl/...
./data/modelnet/cup/...
```

#### Redwood ([Website](http://redwood-data.org/3dscan/dataset.html))
```
./data/redwood/bottle/rgbd/...
./data/redwood/bowl/rgbd/...
./data/redwood/mug/rgbd/...
```

#### RGB-D Object Dataset ([Website](https://rgbd-dataset.cs.washington.edu/index.html))
```
./data/rgbd_objects_uw/{bowl,coffee_mug,shampoo,water_bottle}/{bowl,coffee_mug,shampoo,water_bottle}_1/
./data/rgbd_objects_uw/{bowl,coffee_mug,shampoo,water_bottle}/{bowl,cofee_mug,shampoo,water_bottle}_2/
...
```

## Reproduce Experiments
Depending on which dataset, you have downloaded you can reproduce the results reported in the paper (using the already trained models) by running the script
```bash
source reproduce_{shapenet,modelnet,redwood}_experiments.sh
```
after that, all results can be found in `./results`.

## Train models
To train a network for a specific category you need to first train a per-category VAE, and *afterwards* an initialization network.
### VAE
First we need to convert the ShapeNet meshes to SDFs and optionally filter the dataset. To reproduce the preprocessing of the paper run
```bash
source preprocess_shapenet.sh
```
Then run
```bash
source train_vaes.sh
```
to train the models using the same configuration as used for the paper.

### Init Network
To train the initialization network we used in our paper, run
```bash
source train_init_networks.sh
```
If you want to train the initialization network based on a previously trained object model, you need to create a new config linking to the newly trained VAE. 
See, for example, `sdf_single_shot/configs/discretized_mug.yaml`, which links to `sdf_single_shot/vae_models/mug.yaml`).

## Code Structure
Code is structured into 4 sub-packages:
- *vae*: variational auto-encoder for shapes
- *initialization*: pose and shape estimation from partial pointset
- *differentiable_renderer*: differentiable renderer for discretized SDF
- *estimation*: integration of VAE, differentiable renderer and single shot network into initialization + render and compare pipeline


## `sdfest.differentiable_renderer`
Differentiable rendering of depth image for signed distance fields.

The signed distance field is assumed to be voxelized and it's pose is given by a x, y, z in the camera frame, a quaternion describing its orientation and a scale parameter describing its size. This module provides the derivative with respect to the signed distance values, and the full pose description (position, orientation, scale).

### Generating compile_commands.json
<sup>General workflow for PyTorch extensions (only tested for JIT, probably similar otherwise)</sup>

If you develop PyTorch extensions and want to get correct code checking with ccls / etc. you can do so by going to the ninja build directory (normally `home_directory/.cache/torch_extensions/sdf_renderer_cpp`, or set `load(..., verbose=True)` in `sdf_renderer.py` and check the output), running
```
ninja -t compdb > compile_commands.json
```
and moving `compile_commands.json` to the projects root directory.


## `sdfest.vae`
Architecture to learn a low-dimensional representation of signed-distance fields (i.e., an explicit voxel representation of a signed distance function)

### Installation and training
To run the train script you need to install this package with pip, for example, by running
```bash
pip install -e .
```
inside the root directory. 

You need to preprocess the mesh data prior to running the script, like this:
```bash
python -m sdf_vae.scripts.process_shapenet --inpath shapenet_subfolder --outpath output_path --resolution 64 --padding 2
```
You can control the resolution and added padding so that there is some empty space left in the signed distance field on all borders. If you are running this script via ssh you need to run `export PYOPENGL_PLATFORM=egl` prior to running the script and use the `--all` option which will disable any filtering. Otherwise mesh selection will proceed in two steps: first you see one mesh after another and need to decide which to keep. Pressing left will remove a mesh, pressing right will keep it. After a decision has been made, the conversion will run. Finally another manual selection process is started, where you can remove SDFs in which the mesh to SDF conversion has failed. 

To train the network you can now use either `python -m sdf_vae.scripts.train` or `python sdf_vae/scripts/train.py`.

### config
Configuration can be provided through command-line arguments and hierarchical yaml files. The config files are read in depth first order and later specifications will overwrite previous specifications. 
To summarize:
- command line parameters will take precedence over all config files
- when specifying multiple config files, the second config file will overwrite params set by the first config file
- parent config files overwrite the params set in the contained config files

## `sdfest.estimation`
Modular architecture and experiments for SDF shape and pose estimation

### Usage
You need to manually install the dependencies, which are not on PyPI (see their respective READMEs):
- [YOCO](https://github.com/roym899/yoco)
- [sdf_single_shot](https://github.com/roym899/sdf_single_shot)
- [sdf_differentiable_renderer](https://github.com/roym899/sdf_differentiable_renderer)
- [sdf_vae](https://github.com/roym899/sdf_vae)
- [detectron2](https://github.com/facebookresearch/detectron2)

Then you should be able to run
```bash
pip install git+ssh://git@github.com/roym899/sdf_estimation.git
```

### Development
- Use `pip install -e .` to install the package in editable mode
- Use `pip install -r requirements-dev.txt` to install dev tools
- Use `pytest --cov=sdf_estimation --cov-report term-missing tests/` to run tests and check code coverage

## `sdfest.initialization`
Architectures for single-shot SDF shape and pose estimation from a single (in the future possibly also multiple) depth views.

### Usage
You need to manually install the dependencies, which are not on PyPI (see their respective READMEs):
- [YOCO](https://github.com/roym899/yoco)
- [sdf_differentiable_renderer](https://github.com/roym899/sdf_differentiable_renderer)
- [sdf_vae](https://github.com/roym899/sdf_vae)

Then you should be able to run
```bash
pip install git+ssh://git@github.com/roym899/sdf_single_shot.git
```

To train a new model run
```
python -m sdf_single_shot.scripts.train --config CONFIG_FILE
```
See the `./configs` folder for examples.


### Development
- Use `pip install -e .` to install the package in editable mode
- Use `pip install -r requirements-dev.txt` to install dev tools
- Use `pytest --cov=sdf_single_shot --cov-report term-missing tests/` to run tests and check code coverage
