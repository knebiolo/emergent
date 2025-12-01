"""
Setup script for Emergent package
Agent-based modeling framework for spatial emergent phenomena
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="emergent",
    version="0.1.0",
    author="Kevin Nebiolo",
    author_email="kevin.nebiolo@kleinschmidtgroup.com",
    description="Agent-based modeling framework for ship navigation with real-world environmental forcing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/knebiolo/emergent",  # Update with actual repo URL
    project_urls={
        "Bug Tracker": "https://github.com/knebiolo/emergent/issues",
        "Documentation": "https://github.com/knebiolo/emergent/wiki",
        "Source Code": "https://github.com/knebiolo/emergent",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        # Core scientific stack
        "numpy>=1.20,<2.0",
        "pandas>=1.3,<3.0",
        "scipy>=1.7,<2.0",
        # Geospatial
        "geopandas>=0.9,<1.0",
        "shapely>=1.7,<3.0",
        "fiona>=1.8,<2.0",
        "rasterio>=1.2,<2.0",
        "affine>=2.4,<3.0",
        "pyproj>=3.0,<4.0",
        # Storage / parallel
        "h5py>=3.1,<4.0",
        "dask[array]>=2021.6,<2025.0",
	"numba",
        # OpenGL rendering
        "moderngl>=5.11,<6.0",
        "moderngl-window>=2.4,<3.0",
        "pygame>=2.6,<3.0",
        "pillow>=11.0,<12.0",
        # Ocean/atmospheric data
        "xarray>=0.19,<2024.0",
        "fsspec>=2021.8,<2025.0",
        "s3fs>=2021.8,<2025.0",
        "netcdf4>=1.5,<2.0",
        "h5netcdf>=0.11,<2.0",
        "cfgrib>=0.9.10,<1.0",
        "aiobotocore>=2.0,<3.0",
        "aiohttp>=3.7,<4.0",
        # Utilities
        "networkx>=2.6,<4.0",
        "requests>=2.25,<3.0",
    ],
    extras_require={
        "dev": [
            "black>=22.0",
            "isort>=5.0",
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "salmon": [
            # Additional dependencies for salmon ABM module
            "scikit-learn>=1.0",
            "statsmodels>=0.13",
        ],
    },
    entry_points={
        "console_scripts": [
            "emergent-ship=emergent.ship_abm.simulation_core:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
