import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sdf_estimation",
    version="0.1.0",
    author="Leonard Bruns",
    author_email="roym899@gmail.com",
    description="7-DoF pose and shape estimation architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roym899/sdf_estimation",
    packages=setuptools.find_packages(),
    install_requires=[
        "ffmpeg-python",
        "matplotlib",
        "numpy",
        "open3d",
        "scipy",
        "scikit-image",
        "tabulate",
        "tqdm",
        "torch",
        "sdf_single_shot",
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
