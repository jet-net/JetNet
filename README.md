<p align="center">
  <img width="400" src="https://raw.githubusercontent.com/rkansal47/JetNet/main/docs/_static/images/jetnetlogo.png" />
</p>

<p align="center">
<b>For developing and reproducing ML + HEP projects.</b>
</p>

---

<p align="center">
  <a href="#jetnet">JetNet</a> •
  <a href="#installation">Installation</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#citation">Citation</a> •
  <a href="#references">References</a>
</p>

---

[![CI](https://github.com/jet-net/jetnet/actions/workflows/ci.yml/badge.svg)](https://github.com/jet-net/jetnet/actions)
[![Documentation Status](https://readthedocs.org/projects/jetnet/badge/?version=latest)](https://jetnet.readthedocs.io/en/latest/)
[![Codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/jet-net/JetNet/main.svg)](https://results.pre-commit.ci/latest/github/jet-net/JetNet/main)

[![PyPI Version](https://badge.fury.io/py/jetnet.svg)](https://pypi.org/project/jetnet/)
[![PyPI Downloads](https://pepy.tech/badge/jetnet)](https://pepy.tech/project/jetnet)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10044601.svg)](https://doi.org/10.5281/zenodo.10044601)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05789/status.svg)](https://doi.org/10.21105/joss.05789)

---

## JetNet

JetNet is an effort to increase accessibility and reproducibility in jet-based machine learning.

Currently we provide:

- Easy-to-access and standardised interfaces for the following datasets:
  - [JetNet](https://zenodo.org/record/6975118)
  - [TopTagging](https://zenodo.org/record/2603256)
  - [QuarkGluon](https://zenodo.org/record/3164691)
- Standard implementations of generative evaluation metrics (Ref. [[1, 2](#references)]), including:
  - Fréchet physics distance (FPD)
  - Kernel physics distance (KPD)
  - Wasserstein-1 (W1)
  - Fréchet ParticleNet Distance (FPND)
  - coverage and minimum matching distance (MMD)
- Loss functions:
  - Differentiable implementation of the energy mover's distance [[3](#references)]
- And more general jet utilities.

Additional functionality is under development, and please reach out if you're interested in contributing!

## Installation

JetNet can be installed with pip:

```bash
pip install jetnet
```

To use the differentiable EMD loss `jetnet.losses.EMDLoss`, additional libraries must be installed via

```bash
pip install "jetnet[emdloss]"
```

Finally, [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) must be installed independently for the Fréchet ParticleNet Distance metric `jetnet.evaluation.fpnd` ([Installation instructions](https://github.com/pyg-team/pytorch_geometric#installation)).

## Quickstart

Datasets can be downloaded and accessed quickly, for example:

```python
from jetnet.datasets import JetNet, TopTagging

# as numpy arrays:
particle_data, jet_data = JetNet.getData(
    jet_type=["g", "q"], data_dir="./datasets/jetnet/", download=True
)
# or as a PyTorch dataset:
dataset = TopTagging(
    jet_type="all", data_dir="./datasets/toptagging/", split="train", download=True
)
```

Evaluation metrics can be used as such:

```python
generated_jets = np.random.rand(50000, 30, 3)
fpnd_score = jetnet.evaluation.fpnd(generated_jets, jet_type="g")
```

Loss functions can be initialized and used similarly to standard PyTorch in-built losses such as MSE:

```python
emd_loss = jetnet.losses.EMDLoss(num_particles=30)
loss = emd_loss(real_jets, generated_jets)
loss.backward()
```

## Documentation

The full API reference and tutorials are available at [jetnet.readthedocs.io](https://jetnet.readthedocs.io/en/latest/).
Tutorial notebooks are in the [tutorials](https://github.com/jet-net/JetNet/tree/main/tutorials) folder, with more to come.

<!-- More detailed information about each dataset can (or will) be found at [jet-net.github.io](https://jet-net.github.io/). -->

## Contributing

We welcome feedback and contributions! Please feel free to [create an issue](https://github.com/jet-net/JetNet/issues/new) for bugs or functionality requests, or open [pull requests](https://github.com/jet-net/JetNet/pulls) from your [forked repo](https://docs.github.com/en/get-started/quickstart/fork-a-repo) to solve them.

### Building and testing locally

Perform an editable installation of the package from inside your forked repo and install the `pytest` package for unit testing:

```bash
pip install -e .
pip install pytest
```

Run the test suite to ensure everything is working as expected:

```bash
pytest tests                    # tests all datasets
pytest tests -m "not slow"      # tests only on the JetNet dataset for convenience
```

## Citation

If you use this library for your research, please cite our article in the Journal of Open Source Software:

```
@article{Kansal_JetNet_2023,
  author = {Kansal, Raghav and Pareja, Carlos and Hao, Zichun and Duarte, Javier},
  doi = {10.21105/joss.05789},
  journal = {Journal of Open Source Software},
  number = {90},
  pages = {5789},
  title = {{JetNet: A Python package for accessing open datasets and benchmarking machine learning methods in high energy physics}},
  url = {https://joss.theoj.org/papers/10.21105/joss.05789},
  volume = {8},
  year = {2023}
}
```

Please further cite the following if you use these components of the library.

### JetNet dataset or FPND

```
@inproceedings{Kansal_MPGAN_2021,
  author = {Kansal, Raghav and Duarte, Javier and Su, Hao and Orzari, Breno and Tomei, Thiago and Pierini, Maurizio and Touranakou, Mary and Vlimant, Jean-Roch and Gunopulos, Dimitrios},
  booktitle = "{Advances in Neural Information Processing Systems}",
  editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
  pages = {23858--23871},
  publisher = {Curran Associates, Inc.},
  title = {Particle Cloud Generation with Message Passing Generative Adversarial Networks},
  url = {https://proceedings.neurips.cc/paper_files/paper/2021/file/c8512d142a2d849725f31a9a7a361ab9-Paper.pdf},
  volume = {34},
  year = {2021},
  eprint = {2106.11535},
  archivePrefix = {arXiv},
}
```

### FPD or KPD

```
@article{Kansal_Evaluating_2023,
  author = {Kansal, Raghav and Li, Anni and Duarte, Javier and Chernyavskaya, Nadezda and Pierini, Maurizio and Orzari, Breno and Tomei, Thiago},
  title = {Evaluating generative models in high energy physics},
  reportNumber = "FERMILAB-PUB-22-872-CMS-PPD",
  doi = "10.1103/PhysRevD.107.076017",
  journal = "{Phys. Rev. D}",
  volume = "107",
  number = "7",
  pages = "076017",
  year = "2023",
  eprint = "2211.10295",
  archivePrefix = "arXiv",
}
```

### EMD Loss

Please cite the respective [qpth](https://locuslab.github.io/qpth/) or [cvxpy](https://github.com/cvxpy/cvxpy) libraries, depending on the method used (`qpth` by default), as well as the original EMD paper [[3]](#references).

## References

[1] R. Kansal et al., _Particle Cloud Generation with Message Passing Generative Adversarial Networks_, [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/c8512d142a2d849725f31a9a7a361ab9-Abstract.html) [[2106.11535](https://arxiv.org/abs/2106.11535)].

[2] R. Kansal et al., _Evaluating Generative Models in High Energy Physics_, [Phys. Rev. D **107** (2023) 076017](https://doi.org/10.1103/PhysRevD.107.076017) [[2211.10295](https://arxiv.org/abs/2211.10295)].

[3] P. T. Komiske, E. M. Metodiev, and J. Thaler, _The Metric Space of Collider Events_, [Phys. Rev. Lett. **123** (2019) 041801](https://doi.org/10.1103/PhysRevLett.123.041801) [[1902.02346](https://arxiv.org/abs/1902.02346)].
