# SDF VAE
Architecture to learn a low-dimensional representation of signed-distance fields (i.e., an explicit voxel representation of a signed distance function)

## Installation and training
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

## config
Configuration can be provided through command-line arguments and hierarchical yaml files. The config files are read in depth first order and later specifications will overwrite previous specifications. 
To summarize:
- command line parameters will take precedence over all config files
- when specifying multiple config files, the second config file will overwrite params set by the first config file
- parent config files overwrite the params set in the contained config files
