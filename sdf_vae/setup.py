import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sdf_vae",
    version="0.1.0",
    author="Leonard Bruns",
    author_email="roym899@gmail.com",
    description="Variational autoencoder for generative modelling of signed distance fields",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roym899/sdf_vae",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research"
    ],
    python_requires='>=3.6',
    install_requires=[
        "ffmpeg-python",
        "matplotlib",
        "numpy",
        "PySide2",
        "pyrender",
        "pynput",
        "tensorboard",
        "scikit-image",
        "mesh-to-sdf",
        "tqdm",
        "torch",
        "torchinfo",
        "pynput",
        "trimesh",
        "wandb",
        "sdf_differentiable_renderer",
        "yoco",
    ]
)
