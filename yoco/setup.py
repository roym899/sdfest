import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="yoco",
    version="0.1.1",
    author="Leonard Bruns",
    author_email="roym899@gmail.com",
    description="Minimalistic YAML-based configuration system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roym899/yoco",
    install_requires=["ruamel.yaml"],
    py_modules=["yoco"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research"
    ],
    python_requires='>=3.7',
)
