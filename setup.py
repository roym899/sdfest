import setuptools
import glob
import os

with open("PYPIREADME.md", "r") as fh:
    long_description = fh.read()

config_files = [
    os.path.relpath(path, "sdfest")
    for path in glob.glob("sdfest/**/configs/**/*.yaml", recursive=True)
]

setuptools.setup(
    name="sdfest",
    version="0.1.0",
    author="Leonard Bruns",
    author_email="roym899@gmail.com",
    description="6-DoF pose, scale, and shape estimation architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roym899/sdfest",
    packages=setuptools.find_packages(),
    package_data={
        "sdfest": [
            "differentiable_renderer/csrc/sdf_renderer.cpp",
            "differentiable_renderer/csrc/sdf_renderer_cuda.cu",
        ]
        + config_files
    },
    install_requires=[
        "cpas_toolbox",
        "ffmpeg-python",
        "healpy",
        "joblib",
        "matplotlib",
        "mesh-to-sdf",
        "ninja",
        "numpy",
        "open3d",
        "pandas",
        "pynput",
        "pyrender",
        "PySide2",
        "tensorboard",
        "trimesh",
        "scipy",
        "scikit-image",
        "tabulate",
        "tqdm",
        "torch",
        "torchinfo",
        "torchvision",
        "wandb",
        "yoco",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.7",
)
# include_package_data=True,
# ext_modules=[
#     CUDAExtension(
#         "sdf_renderer_cuda",
#         [
#             "sdfest/differentiable_renderer/csrc/sdf_renderer.cpp",
#             "sdfest/differentiable_renderer/csrc/sdf_renderer_cuda.cu",
#         ],
#     )
# ],
# cmdclass={"build_ext": BuildExtension},
