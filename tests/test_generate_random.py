from collections import Counter

import pytest

from sros.calculation.generate_random import (
    create_original_structure,
    make_supercell_matrix,
    modify_structure_HEDRX,
)


def test_make_supercell_matrix_validates_shape():
    with pytest.raises(ValueError, match="3x3"):
        make_supercell_matrix([[1, 0], [0, 1]])


def test_modify_structure_hedrx_is_reproducible_with_seed():
    structure = create_original_structure()
    structure.make_supercell([[2, 0, 0], [0, 2, 0], [0, 0, 2]])

    first = modify_structure_HEDRX(structure, "TM4", seed=42)
    second = modify_structure_HEDRX(structure, "TM4", seed=42)

    assert [site.specie.symbol for site in first] == [site.specie.symbol for site in second]


def test_modify_structure_hedrx_orders_tm4_species_and_counts_sites():
    structure = create_original_structure()
    structure.make_supercell([[2, 0, 0], [0, 2, 0], [0, 0, 2]])

    modified = modify_structure_HEDRX(structure, "TM4", seed=42)
    symbols = [site.specie.symbol for site in modified]

    assert symbols == sorted(symbols, key=["Li", "Mn", "Ti", "Nb", "O", "F"].index)
    assert Counter(symbols) == {
        "Li": 5,
        "Mn": 2,
        "Nb": 1,
        "O": 7,
        "F": 1,
    }
