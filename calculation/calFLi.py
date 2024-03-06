"""
calFLi.py

This module provides functions to calculate the alpha value for a given crystal structure, specifically for FLi.

Functions:
    - calculate_alpha(structure): Calculate the alpha value for the given crystal structure.
    - additional_function(): Any additional helper function relevant to the calculation.

Example Usage:
    from pymatgen.core.structure import Structure
    from mypackage import calFLi

    # Load crystal structure (replace with actual structure data)
    tm2 = Structure.from_file("D:\\Users\\ASUS\\Desktop\\Computational Practice\\10.27\\TM_FLI-0.325-LLILI0.083\\TM2\\TM2_1_POSCAR")

    # Calculate alpha for the given structure
    alpha_value = calFLi.alpha(tm2, c_cation=26 / 40)
    print("AlphaFLi:", result)

Note:
    This module assumes that the crystal structure is provided in a suitable format compatible with the underlying calculations.
    The calculations are specific to FLi, and adjustments may be needed for other crystal structures.

For more information on the theory and methods used, refer to the relevant literature or documentation.

Author: Liaojh
Date: January 23, 2024
"""

from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.structure import Structure, Element
import numpy as np


def get_idxs(structure: Structure, a: str):
    """
    Obtain index numbers of element a.
    """
    i_idxs = []
    for i in range(structure.num_sites):
        if structure.species[i] == Element(a):
            i_idxs.append(i)
    return i_idxs


def alpha(structure: Structure, c_cation: float):
    """
    Calculate alpha where anion is the i species, cation is the j species, as defined in PNAS, 2021, 118, e2020540118.
    c_cation 是Li的浓度
    """
    F_idxs = get_idxs(structure, "F")
    alpha_list = []
    cnn = CrystalNN()

    for i in F_idxs:
        # P is the probability of finding cation Li adjacent to anion F
        # alpha should be the average value of all the anions F
        P = cnn.get_cn_dict(structure, i).get("Li", 0) / 6
        alpha_list.append(1 - P / c_cation)

    return np.mean(alpha_list)

