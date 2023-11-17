from setuptools import find_packages, setup
from subprocess import check_call
<<<<<<< HEAD
import sys
=======
import sys 
>>>>>>> 0ee589b2d85410cc044279d06bf93e277f182d87


def meta_data():
    meta = {
        "version": "0.1.0",
        "maintainer": "Guillaume Bellec and Christos Sourmpis",
        "email": "christos.sourmpis@epfl.ch; guallaume.bellec@epfl.ch",
        "url": "https://www.epfl.ch/labs/lcn/",
        "license": "MIT",
        "description": "Data fitting with recurrent spiking neural networks (RSNNs) and trial-matching in PyTorch.",
    }

    return meta


def setup_package():
    check_call([sys.executable, "-m", "pip", "install", "numpy==1.21.1"])
    check_call([sys.executable, "-m", "pip", "install", "torch==1.12.1"])
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
            "Topic :: Scientific/Engineering",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
        ],
        packages=find_packages(),
        install_requires=[
            "mat73==0.58",
            "matplotlib==3.8.0",
            "pandas==2.0.2",
            "scikit-learn==0.24.2",
            "scipy==1.8.0",
            "torchvision==0.13.1",
            "tqdm==4.64.1",
            "tsne-torch==1.0.1",
            "numba==0.55.1",
            "umap==0.1.1",
            "umap-learn==0.5.3",
            "geomloss==0.2.5",
            "seaborn==0.12.0",
            "statannot==0.2.3",
            "jupyter==1.0.0",
        ],
    )


if __name__ == "__main__":
    setup_package()
