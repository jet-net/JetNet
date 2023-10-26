<p align="center">
  <img width="400" src="https://raw.githubusercontent.com/rkansal47/JetNet/main/docs/_static/images/jetnetlogo.png" />
</p>

<p align="center">
<b>For developing and reproducing ML + HEP projects.</b>
</p>

______________________________________________________________________

<p align="center">
  <a href="#jetnet">JetNet</a> •
  <a href="#installation">Installation</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#citation">Citation</a> •
  <a href="#references">References</a>
</p>

______________________________________________________________________



[![CI](https://github.com/jet-net/jetnet/actions/workflows/ci.yml/badge.svg)](https://github.com/jet-net/jetnet/actions)
[![Documentation Status](https://readthedocs.org/projects/jetnet/badge/?version=latest)](https://jetnet.readthedocs.io/en/latest/)
[![Codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/jet-net/JetNet/main.svg)](https://results.pre-commit.ci/latest/github/jet-net/JetNet/main)

[![PyPI Version](https://badge.fury.io/py/jetnet.svg)](https://pypi.org/project/jetnet/)
[![PyPI Downloads](https://pepy.tech/badge/jetnet)](https://pepy.tech/project/jetnet)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5598104.svg)](https://doi.org/10.5281/zenodo.5598104)



______________________________________________________________________

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
particle_data, jet_data = JetNet.getData(jet_type=["g", "q"], data_dir="./datasets/jetnet/", download=True)
# or as a PyTorch dataset:
dataset = TopTagging(jet_type="all", data_dir="./datasets/toptagging/", split="train", download=True)
```

Evaluation metrics can be used as such:

```python
generated_jets = np.random.rand(50000, 30, 3)
fpnd_score = jetnet.evaluation.fpnd(generated_jets, jet_type='g')
```

Loss functions can be initialized and used similarly to standard PyTorch in-built losses such as MSE:

```python
emd_loss = jetnet.losses.EMDLoss(num_particles=30)
loss = emd_loss(real_jets, generated_jets)
loss.backward()
```

## Documentation

The full API reference and tutorials are available at [jetnet.readthedocs.io](https://jetnet.readthedocs.io/en/latest/). Tutorial notebooks are in the [tutorials](https://github.com/jet-net/JetNet/tree/main/tutorials) folder, with more to come.

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

If you find this library useful for your research, please consider citing our original paper which introduces it [[1](#references)].

<!--
```latex
@inproceedings{kansal21,
 author = {Raghav Kansal and Javier Duarte and Hao Su and Breno Orzari and Thiago Tomei and Maurizio Pierini and Mary Touranakou and Jean-Roch Vlimant and Dimitrios Gunopulos},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {1--12},
 publisher = {Curran Associates, Inc.},
 title = {Particle Cloud Generation with Message Passing Generative Adversarial Networks},
 url = {https://proceedings.neurips.cc/paper/2020/file/0004d0b59e19461ff126e3a08a814c33-Paper.pdf},
 volume = {33},
 year = {2020}
}
``` -->

Additionally, if you use our EMD loss implementation, please cite the respective [qpth](https://locuslab.github.io/qpth/) or [cvxpy](https://github.com/cvxpy/cvxpy) libraries, depending on the method used (`qpth` by default).


## References

[1] R. Kansal et al., *Particle Cloud Generation with Message Passing Generative Adversarial Networks*, [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/c8512d142a2d849725f31a9a7a361ab9-Abstract.html) [[2106.11535](https://arxiv.org/abs/2106.11535)].

[2] R. Kansal et al., *Evaluating Generative Models in High Energy Physics*, [Phys. Rev. D **107** (2023) 076017](https://doi.org/10.1103/PhysRevD.107.076017) [[2211.10295](https://arxiv.org/abs/2211.10295)].

[3] P. T. Komiske, E. M. Metodiev, and J. Thaler, _The Metric Space of Collider Events_, [Phys. Rev. Lett. __123__ (2019) 041801](https://doi.org/10.1103/PhysRevLett.123.041801) [[1902.02346](https://arxiv.org/abs/1902.02346)].
