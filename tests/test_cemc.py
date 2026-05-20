import random

import numpy as np
from pymatgen.core import Lattice, Structure

from calculation.CEMC import assign_element_numbers, calculate_mn_ratios


def test_calculate_mn_ratios_for_charge_balanced_composition():
    assert calculate_mn_ratios(1.2, 0.4) == (0, 0.4, 0)


def test_calculate_mn_ratios_handles_mixed_mn_valence():
    assert calculate_mn_ratios(1.0, 0.5) == (0.5, 0.0, 0)
    assert calculate_mn_ratios(1.4, 0.5) == (0, 0.0, 0.5)


def test_assign_element_numbers_preserves_site_count_and_symbols():
    random.seed(1)
    structure = Structure(
        Lattice.cubic(4.2),
        ["Li", "Mn", "Mn", "Ti", "O"],
        [
            [0, 0, 0],
            [0.2, 0.2, 0.2],
            [0.4, 0.4, 0.4],
            [0.6, 0.6, 0.6],
            [0.8, 0.8, 0.8],
        ],
    )

    occupancies = assign_element_numbers(structure, 0.5, 0.5, 0)

    assert isinstance(occupancies, np.ndarray)
    assert occupancies.tolist().count(0) == 2
    assert len(occupancies) == structure.num_sites
