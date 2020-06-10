import pathlib
from setuptools import setup, find_packages

setup(
    
    name='beachclassification',
    url='https://github.com/nanidjar/beachclassification',
    author='Niv Anidjar',
    author_email='nanidjar@ucsd.edu',
    
    install_requires=['numpy', 'sklearn', 'scipy', 'laspy', 'skimage', 'utm',
                      'pyyaml','numba','tqdm'],
    packages=find_packages(),
    version='0.0.1',
    
    license='GPL-3.0',
    description='process and label beach LIDAR surveys',
    
    long_description=(pathlib.Path(__file__).parent / "README.md")
    # long_description_content_type = "text/markdown"
)
