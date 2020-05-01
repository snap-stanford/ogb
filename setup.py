from setuptools import setup, find_packages
from os import path
import sys

from io import open

here = path.abspath(path.dirname(__file__))
sys.path.insert(0, path.join(here, 'ogb'))
from version import __version__

print('version')
print(__version__)

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

package_data_list = ['ogb/graphproppred/master.csv', 'ogb/nodeproppred/master.csv', 'ogb/linkproppred/master.csv']

setup(name='ogb',
      version=__version__,
      description='Open Graph Benchmark',
      url='https://github.com/snap-stanford/ogb',
      author='OGB Team',
      author_email='ogb@cs.stanford.edu',
      keywords=['pytorch', 'graph machine learning', 'graph representation learning', 'graph neural networks'],
      long_description=long_description,
      long_description_content_type='text/markdown',
      install_requires = [
        'torch>=1.2.0',
        'numpy>=1.16.0',
        'tqdm>=4.29.0',
        'scikit-learn>=0.20.0',
        'pandas>=0.24.0',
        'six>=1.12.0',
        'urllib3>=1.24.0'
      ],
      license='MIT',
      packages=find_packages(exclude=['dataset', 'examples', 'docs']),
      package_data={'ogb': package_data_list},
      include_package_data=True,
      classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
    ],
)
