<p align="center"><img src="docs/logo2.png" width="500px" alt=" "></p>

<h1 align="center">Short-range Ordering based Swapping method</h1>

<h4 align="center">

[![Static Badge](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![status](https://joss.theoj.org/papers/e96a568ca53ee9d14548d7b8bed69b25/status.svg)](https://joss.theoj.org/papers/e96a568ca53ee9d14548d7b8bed69b25)

</h4>

*Lightweight but caffeinated Python implementation of computational methods
for construction of high-entropy disordered rocksalt cathode materials.*

-----------------------------------------------------------------------------

**SROS** is an efficiently short-range ordering based swapping (SROS) method to construct a HE-DRX model by combining density functional theory (DFT) calculations and Monte Carlo (MC) simulations. Specifically, **SROS** is a fast and efficient modeling approach to construct specific SRO in DRX structures.

Functionality
-------------
**SROS** currently includes the following functionality:

-   Defining cluster expansion functions for a given disordered structure using a
    variety of available site basis functions with and without explicit
    redundancy.

-   Option to include explicit electrostatics in expansions using the Ewald summation
    method.
-   Computing correlation vectors for a set of training structures with a variety
    of functionality to inspect the resulting feature matrix.

-   Defining fitted cluster expansions for subsequent property prediction.
-   Fast evaluation of correlation vectors and differences in correlation vectors
    from local updates in order to quickly compute properties and changes in
    properties for specified supercell sizes.

-   Flexible toolset to sample cluster expansions using Monte Carlo with
    canonical, semigrand canonical, and charge neutral semigrand canonical ensembles
    using a Metropolis or a Wang-Landau sampler.

-   Special quasi-random structure generation based on either correlation vectors or
    cluster interaction vectors.

-   Solving for periodic ground-states of any given cluster expansion with or without
    electrostatics over a given supercell.

**SROS** is built on top of [pymatgen](https://pymatgen.org) so any pre/post
structure analysis can be done seamlessly using the various functionality
supported there.

Installation
------------
From source:

`Clone` the repository. The latest tag in the `main` branch is the stable version of the
code. The `main` branch has the newest tested features, but may have more
lingering bugs. From the top level directory

    pip install .

The only known installation issue
is building `pymatgen` dependencies. If running `pip install .` fails, try
installing `pymatgen` with conda first:

    conda install -c conda-forge pymatgen

Citing
------
If you use **SROS** in your research, please give the repo a star :star:

Contributing
------------
We welcome all your contributions with open arms! Please fork and pull request any contributions.


