<p align="center"><img src="docs/logo2.png" width="500px" alt="SROS Logo"></p>

<h1 align="center">Short-Range Order Swapping (SROS) Method</h1>

<h4 align="center">

[![Static Badge](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](http://img.shields.io/badge/DOI-10.1002/aenm.202501857-B31B1B.svg)](https://doi.org/10.1002/aenm.202501857)

</h4>

*Efficient atomistic modeling of Short-Range Order in High-Entropy Cation-Disordered Rocksalt-Type Cathodes.*

-----------------------------------------------------------------------------

**SROS** (Short-Range Order Swapping) is a Python-based computational framework developed to construct atomistic models for high-entropy (HE) ceramics, with a specific focus on **Cation-Disordered Rocksalt (DRX)** cathodes. 

Unlike conventional approaches that involve computationally expensive many-body interaction fitting (e.g., Cluster Expansion), **SROS** utilizes a **descriptor-based swapping algorithm**. By leveraging **Warren-Cowley SRO parameters** as key descriptors, it efficiently generates structures that accurately represent experimental features (validated by neutron pair distribution function data) while significantly reducing computational costs.

## Key Features & Functionality

**SROS** provides a robust workflow for modeling structural complexity in multicomponent systems:

*   **Descriptor-Based SRO Construction**: 
    *   Efficiently constructs atomistic models with specific Short-Range Order (SRO) targets.
    *   Quantifies local ordering using **Warren–Cowley (WC) parameters** ($\alpha_{ij}^{(n)}$) for both cation and anion sublattices.
    *   Supports complex high-entropy compositions (e.g., quinary systems like TM4/TM6).

*   **Targeted Coordination Tuning**:
    *   **1NN Environment**: Rational construction of first-nearest-neighbor clusters (e.g., optimizing $\alpha_{Li-F}^{(1)}$ for Li-F rich environments).
    *   **2NN Environment**: Tuning of second-nearest-neighbor correlations (e.g., controlling cation clustering $\alpha_{Li-Li}^{(2)}$).

*   **Electrostatic Optimization**: 
    *   Integrates **Simulated Annealing** with **Ewald Summation** to minimize Coulomb electrostatic interaction energy, ensuring thermodynamically reasonable configurations.

*   **Transport & Percolation Analysis**:
    *   Solves site percolation problems to evaluate Li-ion diffusion pathways (e.g., 0-TM channels).
    *   Analyzes the impact of configurational entropy on percolating networks.

*   **Versatility**: 
    *   While optimized for DRX oxyfluorides ($Li_{1+x}TM_{1-x}O_{2-y}F_y$), the method is adaptable to other crystal systems characterized by long-range disorder, such as **Garnet-type oxides** (e.g., LLZTO).

**SROS** is built on top of [pymatgen](https://pymatgen.org), allowing seamless integration with high-throughput materials analysis workflows.

## Citing

If you use **SROS** in your research, please consider citing our paper:

**Liao, J., Chen, H., Xie, Y., Li, Z., Tan, S., Zhou, S., Jiang, L., Zhang, X., Liu, M., He, Y.-B., Kang, F., Lun, Z., Zhao, S., Hou, T.** (2025). *Modeling Short-Range Order in High-Entropy Cation-Disordered Rocksalt-Type Cathodes*. **Advanced Energy Materials**, 2501857. 

[![DOI](https://img.shields.io/badge/DOI-10.1002/aenm.202501857-blue)](https://doi.org/10.1002/aenm.202501857)
[![PDF](https://img.shields.io/badge/Download-PDF-red)](./docs/AdvancedEnergyMaterials-2025-Liao-Modeling_Short‐Range_Order_in_High‐Entropy_Cation‐Disordered_Rocksalt‐Type.pdf)

```bibtex
@article{Liao2025SROS,
  title = {Modeling Short-Range Order in High-Entropy Cation-Disordered Rocksalt-Type Cathodes},
  author = {Liao, Junhong and Chen, Hao and Xie, Yaoshu and Li, Zihui and Tan, Shendong and Zhou, Shuyu and Jiang, Lu and Zhang, Xiang and Liu, Ming and He, Yan-Bing and Kang, Feiyu and Lun, Zhengyan and Zhao, Shixi and Hou, Tingzheng},
  journal = {Advanced Energy Materials},
  year = {2025},
  volume = {},
  pages = {2501857},
  doi = {10.1002/aenm.202501857}
}

Contributing
------------
We welcome all your contributions with open arms! Please fork and pull request any contributions.


