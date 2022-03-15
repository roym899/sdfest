import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sdfest",
    version="0.1.0",
    author="Leonard Bruns",
    author_email="roym899@gmail.com",
    description="7-DoF pose and shape estimation architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roym899/sdfest",
    packages=setuptools.find_packages(),
    package_data={
        'sdf_differentiable_renderer': [
            'differentiable_renderer/csrc/sdf_renderer.cpp',
            'differentiable_renderer/csrc/sdf_renderer_cuda.cu'
        ]
    },
    install_requires=[
        "ffmpeg-python",
        "healpy",
        "joblib",
        "matplotlib",
        "numpy",
        "open3d",
        "pandas",
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
    #             "sdf_differentiable_renderer/src/sdf_renderer.cpp",
    #             "sdf_differentiable_renderer/src/sdf_renderer_cuda.cu",
    #         ],
    #     )
    # ],
    # cmdclass={"build_ext": BuildExtension},
)
