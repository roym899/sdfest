import setuptools
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sdf_differentiable_renderer",
    version="0.2.0",
    author="Leonard Bruns",
    author_email="roym899@gmail.com",
    description="CUDA-based differentiable renderer for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roym899/sdf_differentiable_rendering",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research"
    ],
    packages=setuptools.find_packages(),
    package_data={'sdf_differentiable_renderer': ['src/sdf_renderer.cpp',
                                                  'src/sdf_renderer_cuda.cu']},
    install_requires=[
        "torch",
        "matplotlib",
        "scipy",
        "numpy",
        "open3d",
        "tqdm",
        "click",
    ]
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
