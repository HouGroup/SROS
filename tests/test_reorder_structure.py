import pytest
from pymatgen.core import Lattice, Structure

from sros.calculation.reorder_structure import reorder_atoms, reorder_atoms_flexible


def test_reorder_atoms_preserves_all_site_property_keys():
    structure = Structure(
        Lattice.cubic(4.2),
        ["O", "Li", "Mn"],
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
    )
    structure[0].properties["label"] = "anion"
    structure[1].properties["charge"] = 1
    structure[2].properties["label"] = "transition-metal"
    structure[2].properties["charge"] = 3

    reordered = reorder_atoms(structure, ["Li", "Mn", "O"])

    assert [site.specie.symbol for site in reordered] == ["Li", "Mn", "O"]
    assert reordered.site_properties["label"] == [None, "transition-metal", "anion"]
    assert reordered.site_properties["charge"] == [1, 3, None]


def test_reorder_atoms_raises_for_missing_elements():
    structure = Structure(
        Lattice.cubic(4.2),
        ["Li", "O", "F"],
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
    )

    with pytest.raises(ValueError, match="not in order list"):
        reorder_atoms(structure, ["Li", "O"])


def test_reorder_atoms_flexible_appends_unlisted_elements_alphabetically():
    structure = Structure(
        Lattice.cubic(4.2),
        ["F", "Nb", "Li", "O", "Mn"],
        [
            [0, 0, 0],
            [0.2, 0.2, 0.2],
            [0.4, 0.4, 0.4],
            [0.6, 0.6, 0.6],
            [0.8, 0.8, 0.8],
        ],
    )

    reordered = reorder_atoms_flexible(structure, ["Li", "Mn", "O"])

    assert [site.specie.symbol for site in reordered] == ["Li", "Mn", "O", "F", "Nb"]
