---
title: 'JetNet: A Python package for accessing open datasets and benchmarking machine learning methods in high energy physics'
tags:
  - Python
  - PyTorch
  - high energy physics
  - machine learning
  - jets
authors:
  - name: Raghav Kansal
    orcid: 0000-0003-2445-1060
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
    corresponding: true
  - name: Carlos Pareja
    orcid: 0000-0002-9022-2349
    affiliation: 1
  - name: Zichun Hao
    orcid: 0000-0002-5624-4907
    affiliation: 3
  - name: Javier Duarte
    orcid: 0000-0002-5076-7096
    affiliation: 1
affiliations:
 - name: UC San Diego, USA
   index: 1
 - name: Fermilab, USA
   index: 2
 - name: California Institute of Technology, USA
   index: 2
date: 2023
bibliography: paper.bib
---

# Summary

`JetNet` is a Python package that aims to increase accessibility and reproducibility for machine learning (ML) research in high energy physics (HEP), primarily related to particle jets. Based on the popular PyTorch ML framework, it provides easy-to-access and standardized interfaces for multiple heterogeneous HEP datasets and implementations of evaluation metrics, loss functions, and more general utilities relevant to HEP.


# Statement of need

It is essential in scientific research to maintain standardized benchmark datasets following the findable, accessible, interoperable, and reproducible (FAIR) data principles (see @Chen:2021euv), practices for using the data, and methods for evaluating and comparing different algorithms. This can often be difficult in high energy physics (HEP) because of the broad set of formats in which data is released and the expert knowledge required to parse the relevant information. The `JetNet` Python package aims to facilitate this by providing a standard interface and format for HEP datasets, integrated with PyTorch [@NEURIPS2019_9015], to improve accessibility for both HEP experts and new or interdisciplinary researchers looking to do ML. Furthermore, by providing standard formats and implementations for evaluation metrics, results are more easily reproducible, and models are more easily assessed and benchmarked. `JetNet` is complementary to existing efforts for improving HEP dataset accessibility, notably the `EnergyFlow` library [@Komiske:2019jim], with a unique focus to ML applications and integration with PyTorch.


## Content

`JetNet` currently provides easy-to-access and standardized interfaces for the JetNet [@kansal_raghav_2022_6975118], top quark tagging [@kasieczka_gregor_2019_2603256; @Kasieczka:2019dbj], and quark-gluon tagging [@komiske_patrick_2019_3164691] reference datasets, all hosted on Zenodo [@Zenodo]. It also provides standard implementations of generative evaluation metrics [@Kansal:2021cqp; @Kansal:2022spb], including Fréchet physics distance (FPD), kernel physics distance (KPD), 1-Wasserstein distance (W1), Fréchet ParticleNet distance (FPND), coverage, and minimum matching distance (MMD). Finally, `JetNet` implements custom loss functions like a differentiable version of the energy mover's distance [@PhysRevLett.123.041801] and more general jet utilities.


## Impact

The impact of `JetNet` is demonstrated by the surge in ML and HEP research facilitated by the package, including in the areas of generative adversarial networks [@Kansal:2021cqp], transformers [@Kach:2022uzq; @Kansal:2022spb; @Kach:2023rqw], diffusion models [@Leigh:2023toe; @Mikuni:2023dvk], and equivariant networks [@Hao:2022zns; @Buhmann:2023pmh], all accessing datasets, metrics, and more through `JetNet`.


## Future Work

Future work will expand the package to additional dataset loaders, including detector-level data, and different machine learning backends such as JAX [@jax2018github]. Improvements to the performance, such as optional lazy loading of large datasets, are also planned, as well as community challenges to benchmark algorithms facilitated by `JetNet`.


# Acknowledgements

We thank the `JetNet` community for their support and feedback. J.D. and R.K. received support for work related to `JetNet` provided by the U.S. Department of Energy (DOE), Office of Science, Office of High Energy Physics Early Career Research Program under Award No. DE-SC0021187, the DOE, Office of Advanced Scientific Computing Research under Award No. DE-SC0021396 (FAIR4HEP). R.K. was partially supported by the LHC Physics Center at Fermi National Accelerator Laboratory, managed and operated by Fermi Research Alliance, LLC under Contract No. DE-AC02-07CH11359 with the DOE. C.P. was supported by the Experiential Projects for Accelerated Networking and Development (EXPAND) mentorship program at UC San Diego.


# References
