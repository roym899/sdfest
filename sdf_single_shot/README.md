# SDF Single Shot
Architectures for single-shot SDF shape and pose estimation from a single (in the future possibly also multiple) depth views.

## Usage
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


## Development
- Use `pip install -e .` to install the package in editable mode
- Use `pip install -r requirements-dev.txt` to install dev tools
- Use `pytest --cov=sdf_single_shot --cov-report term-missing tests/` to run tests and check code coverage
