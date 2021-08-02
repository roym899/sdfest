# SDF Differentiable Renderer 
Differentiable rendering of depth image for signed distance fields.

The signed distance field is assumed to be voxelized and it's pose is given by a x, y, z in the camera frame, a quaternion describing its orientation and a scale parameter describing its size. This module provides the derivative with respect to the signed distance values, and the full pose descroption (position, orientation, scale).

## Generating compile_commands.json
<sup>General workflow for PyTorch extensions (only tested for JIT, probably similar otherwise)</sup>

If you develop PyTorch extensions and want to get correct code checking with ccls / etc. you can do so by going to the ninja build directory (normally `home_directory/.cache/torch_extensions/sdf_renderer_cpp`, or set `load(..., verbose=True)` in `sdf_renderer.py` and check the output), running
```
ninja -t compdb > compile_commands.json
```
and moving `compile_commands.json` to the projects root directory.

