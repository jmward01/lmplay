"""
Setup configuration for the lmplay framework.

This module configures the lmplay package for installation, including:
- Package discovery and inclusion
- Dependency management  
- CLI command entry points
- Version and author information

The lmplay framework provides a modular system for rapid experimentation
with language model architectures, training pipelines, and analysis tools.
"""

from setuptools import setup, find_namespace_packages

setup(name='lmplay',
      version='0.1',
      author='Jeff Ward',
      packages=find_namespace_packages(include=["lmplay*"]),
      install_requires=['datasets',
                        'levenshtein',
                        'torch',
                        'tqdm',
                        'matplotlib',
                        'numpy',
                        'transformers',
                        'tiktoken'],
      extras_require={
          'test': [
              'pytest>=7.0.0',
              'pytest-cov>=3.0.0',
          ]
      },
      entry_points={'console_scripts': ['lmp_trainer=lmplay.train.__main__:main',
                                        'lmp_generator=lmplay.generate.__main__:main',
                                        'lmp_plotstats=lmplay.stats.__main__:main',
                                        'lmp_cleanmodel=lmplay.cleanmodel.__main__:main']})
