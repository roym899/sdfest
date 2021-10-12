import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sdf_single_shot",
    version="0.1.0",
    author="Leonard Bruns",
    author_email="roym899@gmail.com",
    description="7-DoF pose estimation of signed distance fields",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roym899/sdf_single_shot",
    packages=setuptools.find_packages(),
    install_requires=[
        "healpy",
        "wandb",
        "torch",
        "torchvision",
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "torchinfo",
        "yoco",
        "sdf_differentiable_renderer",
        "sdf_vae",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.6",
)
