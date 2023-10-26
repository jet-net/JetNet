import re

from setuptools import find_packages, setup

install_requires = [
    "numpy >= 1.21.0",
    "torch >= 1.8.0",
    "energyflow >= 1.3.0",
    "scipy >= 1.6.2",
    "awkward >= 1.4.0",
    "coffea >= 0.7.0",
    "h5py >= 3.0.0",
    "pandas",
    "tables",
    "requests",
    "tqdm",
]

extras_require = {"emdloss": ["qpth", "cvxpy"]}


classifiers = [
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    "Development Status :: 3 - Alpha",
    # Pick your license as you wish (should match "license" above)
    "License :: OSI Approved :: MIT License",
    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]


def readme():
    with open("README.md") as f:
        return f.read()


with open("jetnet/__init__.py", "r") as f:
    __version__ = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read()).group(1)


setup(
    name="jetnet",
    version=__version__,
    description="Jets + ML integration",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="http://github.com/jet-net/JetNet",
    author="Raghav Kansal",
    author_email="rkansal@cern.ch",
    license="MIT",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.7",
    extras_require=extras_require,
    classifiers=classifiers,
    zip_safe=False,
    include_package_data=True,
)
