from setuptools import setup, find_packages
import re

install_requires = [
    "numpy >= 1.21.0",
    "torch >= 1.8.0",
    "energyflow >= 1.3.0",
    "scipy >= 1.6.2",
    "awkward >= 1.4.0",
    "coffea >= 0.7.0",
    "requests",
    "tqdm",
]

extras_require = {"emdloss": ["qpth", "cvxpy"]}


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
    url="http://github.com/rkansal47/JetNet",
    author="Raghav Kansal",
    author_email="rkansal@cern.ch",
    license="MIT",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
    zip_safe=False,
    include_package_data=True,
)
