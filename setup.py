from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='sentinel2',
      version='0.14',
      description='simple library for handling sentinel 2 requests and images',
      url='http://github.com/mt-krainski/lithopia',
      author='M. Krainski',
      author_email='mateusz@krainski.eu',
      license='MIT',
      install_requires=required,
      packages=['sentinel2'],
      zip_safe=False)
