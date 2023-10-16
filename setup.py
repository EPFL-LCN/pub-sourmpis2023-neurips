from itertools import dropwhile
from os import path
from setuptools import find_packages, setup
from subprocess import DEVNULL, call


def collect_docstring(lines):
    """Return document docstring if it exists"""
    lines = dropwhile(lambda x: not x.startswith('"""'), lines)
    doc = ""
    for line in lines:
        doc += line
        if doc.endswith('"""\n'):
            break

    return doc[3:-4].replace("\r", "").replace("\n", " ")


def meta_data():
    meta = {
        "version": "0.0.1",
        "maintainer": "Guillaume Bellec and Christos Sourmpis",
        "email": "guillaume@bellec.eu",
        "url": "https://www.epfl.ch/labs/lcn/",
        "license": "Apache 2.0",
        "description": "Efficient implementation of recurrent spiking neural networks (RSNNs) in PyTorch.",
    }

    return meta


def setup_package():
    with open("README.md") as f:
        long_description = f.read()
    meta = meta_data()
    setup(
        name="rsnn",
        version=meta["version"],
        description=meta["description"],
        long_description=long_description,
        long_description_content_type="text/markdown",
        maintainer=meta["maintainer"],
        maintainer_email=meta["email"],
        url=meta["url"],
        license=meta["license"],
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: Apache 2.0",
            "Topic :: Scientific/Engineering",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
        ],
        packages=find_packages(),
        install_requires=[
            "mat73==0.58",
            "matplotlib==3.8.0",
            "numpy==1.24.0",
            "pandas==2.0.2",
            "scikit-learn==0.24.2",
            "scipy==1.8.0",
            "torch==1.12.1",
            "torchvision==0.13.1",
            "tqdm==4.64.1",
            "tsne-torch==1.0.1",
            "umap==0.1.1",
            "umap-learn==0.5.3",
            "geomloss==0.2.5",
            "seaborn==0.12.0",
            "statannot==0.2.3",
        ],
    )


if __name__ == "__main__":
    setup_package()
