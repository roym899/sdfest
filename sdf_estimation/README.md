# SDF Estimation
Modular architecture and experiments for SDF shape and pose estimation

## Usage
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

## Development
- Use `pip install -e .` to install the package in editable mode
- Use `pip install -r requirements-dev.txt` to install dev tools
- Use `pytest --cov=sdf_estimation --cov-report term-missing tests/` to run tests and check code coverage
