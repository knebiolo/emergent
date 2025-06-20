# -*- coding: utf-8 -*-

# <project_root>/emergent/setup.py
from setuptools import setup, find_packages

setup(
    name="emergent",
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.3",
        "geopandas>=0.9",
        "shapely>=1.7",
        "fiona>=1.8",
        "requests>=2.25",
        "pyqt5>=5.15",
        "pyqtgraph>=0.12",
        "networkx>=2.6"
        ]

)
