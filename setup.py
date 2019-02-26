from setuptools import setup

setup(name='threediqtt',
      version='1.0',
      install_requires=['numpy',
                        'matplotlib',
                        'torch', # mainly because they have a nice data loader class
                        'h5py']
      )
