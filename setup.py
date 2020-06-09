import pathlib
from setuptools import setup, find_packages

setup(
    
    name='beach_classification',
    url='https://github.com/nanidjar/beach_classification',
    author='Niv Anidjar',
    author_email='nanidjar@ucsd.edu',
    
    install_requires=['numpy', 'sklearn', 'os', 'pickle', 'scipy', 'laspy', 'skimage', 'utm',
                      'pyyaml','numba','tqdm'],
    packages=find_packages(),
    version='0.0.1',
    
    license='GPL-3.0',
    description='process and label beach LIDAR surveys',
    
    long_description=open("README.md").read()
)
