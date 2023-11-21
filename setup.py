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
      entry_points={'console_scripts': ['lmp_trainer=lmplay.train.__main__:main',
                                        'lmp_generator=lmplay.generate.__main__:main',
                                        'lmp_plotstats=lmplay.stats.__main__:main',
                                        'lmp_cleanmodel=lmplay.cleanmodel.__main__:main']})
