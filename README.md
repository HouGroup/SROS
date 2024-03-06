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

-   Special quasi-random structure generation based on either correlation vectors or cluster interaction vectors.

-   Solving the site percolation problem of percolation theory for a given set of percolation rules. These rules can be quite complex and reflect the physical interactions of the percolating species with other atomic species in the structure.

-   Constructing a rational first-nearest-neighbor (1NN) coordination environment, quantified by the Warren–Cowley SRO parameter αFLi.

-   Constructing a rational second-nearest-neighbor (2NN) coordination environment, quantified by the Warren–Cowley SRO parameter αLiLi.

-   Achieving a more thermodynamically stable configuration by lowering the Coulomb electrostatic interaction energy, which is calculated using the Ewald Summation method. 




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


