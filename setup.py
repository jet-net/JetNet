from setuptools import setup, find_packages

install_requires = [
    'numpy >= 1.21.0',
    'torch >= 1.8.0',
    'energyflow >= 1.3.0',
    'scipy >= 1.6.2',
    'awkward >= 1.4.0',
    'coffea >= 0.7.0',
    'requests',
    'tqdm',
]


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='jetnet',
      version='0.0.1.post2',
      description='One jet library to rule them all.',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='http://github.com/rkansal47/JetNet',
      author='Raghav Kansal',
      author_email='rkansal@cern.ch',
      license='MIT',
      packages=find_packages(),
      install_requires=install_requires,
      zip_safe=False)
