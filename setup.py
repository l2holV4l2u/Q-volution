from setuptools import setup, find_packages

setup(
    name="dc_qaoa",
    version="0.1.0",
    description="Divide-and-conquer QAOA pipeline for weighted Max-Cut",
    author="",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0",
        "pyarrow>=14.0",
        "networkx>=3.2",
        "numpy>=1.26",
        "scipy>=1.12",
    ],
    entry_points={
        "console_scripts": [
            "dc_qaoa=dc_qaoa.main:main",
        ],
    },
    python_requires=">=3.11",
)
