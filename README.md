# JetNet


A library for developing and reproducing jet-based machine learning (ML) projects.

JetNet provides common standardized PyTorch-based datasets, evaluation metrics, and loss functions for working with jets using ML. Currently supports the flagship JetNet dataset, and the Fréchet ParticleNet Distance (FPND), Wasserstein-1 (W1), coverage and minimum matching distance (MMD) metrics all introduced in Ref. [[1](#References)], as well as jet utilities and differentiable implementation of the energy mover's distance [[2](#References)] for use as a loss function. Additional functionality is currently under development.


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

Datasets can be loaded quickly with, for example:

```python
dataset = jetnet.datasets.JetNet(jet_type='g')
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

Full API reference is available at [jetnet.readthedocs.io](https://jetnet.readthedocs.io/en/latest/).

More detailed information about each dataset can (or will) be found at [jet-net.github.io](https://jet-net.github.io/).

*Tutorials for datasets and functions are coming soon.*


### References

[1] R. Kansal et al. *Particle Cloud Generation with Message Passing Generative Adversarial Networks* (2021) [[2106.11535](https://arxiv.org/abs/2106.11535)]

[2] P. T. Komiske, E. M. Metodiev, and J. Thaler, _The Metric Space of Collider Events_, [Phys. Rev. Lett. __123__ (2019) 041801](https://doi.org/10.1103/PhysRevLett.123.041801) [[1902.02346](https://arxiv.org/abs/1902.02346)].
