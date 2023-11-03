import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


requirements = [
    "arrow",
    "binaryornot",
    "certifi",
    "chardet",
    "charset-normalizer",
    "click",
    "contourpy",
    "cookiecutter",
    "cycler",
    "filelock",
    "fonttools",
    "fsspec",
    "idna",
    "importlib-resources",
    "Jinja2",
    "joblib",
    "kiwisolver",
    "littleutils",
    "markdown-it-py",
    "MarkupSafe",
    "matplotlib",
    "mdurl",
    "mpmath",
    "networkx",
    "numpy",
    "ogb",
    "outdated",
    "packaging",
    "pandas",
    "Pillow",
    "pip-chill",
    "psutil",
    "Pygments",
    "pyparsing",
    "python-dateutil",
    "python-slugify",
    "pytz",
    "PyYAML",
    "requests",
    "rich",
    "scikit-learn",
    "scipy",
    "six",
    "sympy",
    "text-unidecode",
    "threadpoolctl",
    "torch",
    "torch-scatter",
    "torch-sparse",
    "torch_geometric",
    "tqdm",
    "types-python-dateutil",
    "typing_extensions",
    "tzdata",
    "urllib3",
    "zipp"
]


setup(
    name="fotom",
    version="0.0.1",
    url="https://github.com/neutralpronoun/fotom",
    author="neutralpronoun",
    author_email="alexander.davies@bristol.ac.uk",
    description="A pip package for FoToM (Foundational Topology Models), pretrained models for graph deep learning",
    long_description="https://github.com/neutralpronoun/fotom",
    packages=find_packages(exclude=("tests",)),
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6"
    ],
    package_data={'fotom':["fotom.pt", "fotom.yaml"]},
    test_suite = "pytest",
    tests_retuire=["pytest"]
)
