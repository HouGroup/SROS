"""
Tools to build NaCl-based supercells and substitute them into DRX-like
Li/TM/O/F structures, including:

- simple random substitution with fixed TM2/TM4/TM6 rules
- composition-controlled Li/Mn/Ti/O substitution
- SQS generation helpers (SMOL / ATAT)

"""

import random
from collections import defaultdict

import numpy as np
from pymatgen.core import Lattice, Structure, Element
from smol.capp.generate.special.sqs import StochasticSQSGenerator
from pymatgen.command_line.mcsqs_caller import run_mcsqs

from .reorder_structure import reorder_atoms

# ----------------------------------------------------------------------
# Basic NaCl prototype and supercell construction
# ----------------------------------------------------------------------

def create_original_structure() -> Structure:
    """
    build NaCl cells: Fm-3m, Na at (0,0,0), Cl at (0.5,0.5,0.5).
    """
    lattice = [
        [0.0, 2.1, 2.1],
        [2.1, 0.0, 2.1],
        [2.1, 2.1, 0.0],
    ]

    species = ["Na", "Cl"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
    ]

    return Structure(lattice, species, coords)



def make_supercell_matrix(scaling_matrix=None) -> Structure:
    """
    使用一般 3x3 整数矩阵构造 NaCl 超胞。

    等价于原 `generate_random_any_composition.make_supercell`。
    """
    if scaling_matrix is None:
        scaling_matrix = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]

    original = create_original_structure()
    scaling_matrix = np.array(scaling_matrix, dtype=int)

    if scaling_matrix.shape != (3, 3):
        raise ValueError("Scaling matrix must be a 3x3 integer matrix")

    supercell = original.copy()
    supercell.make_supercell(scaling_matrix)
    return supercell


# ----------------------------------------------------------------------
# Random substitution of Li/Mn/Ti/O with controlled components
# ----------------------------------------------------------------------

def modify_structure(li_content: float, mn_content: float, input_structure: Structure) -> Structure:
    """
    按给定 Li / Mn 含量，对 NaCl 超胞进行随机替换，得到 Li/Mn/Ti/O 结构。

    - Na → Li
    - Cl → O
    - 部分 Li → Mn
    - 剩余 Li → Ti
    """
    structure = input_structure.copy()

    # Na → Li
    na_sites = [i for i, site in enumerate(structure) if site.species_string == "Na"]
    for i in na_sites:
        structure[i] = Element("Li")

    # Cl → O
    cl_sites = [i for i, site in enumerate(structure) if site.species_string == "Cl"]
    for i in cl_sites:
        structure[i] = Element("O")

    ti_content = 2 - li_content - mn_content

    li_sites = [i for i, site in enumerate(structure) if site.species_string == "Li"]
    num_mn = int(round(len(li_sites) * mn_content / 2))
    selected_mn = random.sample(li_sites, num_mn)
    for i in selected_mn:
        structure[i] = Element("Mn")
    remaining_li = [idx for idx in li_sites if idx not in selected_mn]
    num_ti = int(round(len(li_sites) * ti_content / 2))
    selected_ti = random.sample(remaining_li, num_ti)
    for i in selected_ti:
        structure[i] = Element("Ti")

    ordered_structure = reorder_atoms(structure, ["Li", "Mn", "Ti", "O"])
    return ordered_structure


# ----------------------------------------------------------------------
# The TM2/TM4/TM6 rule substitution of Na/Cl → Li/TM/O/F
# ----------------------------------------------------------------------

def modify_structure_HEDRX(structure: Structure, tm_type: str, seed: int | None = None) -> Structure:
    """
    On the NaCl structure, Na/Cl -> Li/TM/O/F was replaced according to the TM2/TM4/TM6 scheme.
    """
    # 1. Define the replacement rule and add the "order" key for subsequent sorting
    substitution_rules = {
        "TM2": {
            "Na": [("Li", 0.65), ("Mn", 0.20), ("Ti", 0.15)],
            "Cl": [("O", 0.85), ("F", 0.15)],
            "order": ["Li", "Mn", "Ti", "O", "F"]
        },
        "TM4": {
            "Na": [("Li", 0.65), ("Mn", 0.20), ("Ti", 0.05), ("Nb", 0.10)],
            "Cl": [("O", 0.85), ("F", 0.15)],
            "order": ["Li", "Mn", "Ti", "Nb", "O", "F"]
        },
        "TM6": {
            "Na": [
                ("Li", 0.65), ("Mn", 0.10), ("Co", 0.05), ("Cr", 0.05), ("Ti", 0.05), ("Nb", 0.10),
            ],
            "Cl": [("O", 0.85), ("F", 0.15)],
            "order": ["Li", "Mn", "Co", "Cr", "Ti", "Nb", "O", "F"]
        },
    }

    if tm_type not in substitution_rules:
        raise ValueError(f"Unsupported tm_type: {tm_type}. Use 'TM2', 'TM4', or 'TM6'.")

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    new_structure = structure.copy()

    na_sites = [i for i, site in enumerate(new_structure) if site.species_string == "Na"]
    cl_sites = [i for i, site in enumerate(new_structure) if site.species_string == "Cl"]

    if not na_sites or not cl_sites:
        raise ValueError("Input structure does not contain 'Na' or 'Cl' sites. "
                         "Check if the structure was already modified.")

    def _substitute_sites(site_indices, substitution_list):
        np.random.shuffle(site_indices)
        site_count = len(site_indices)
        substitutions = []

        counts = [int(round(site_count * fraction)) for _, fraction in substitution_list]
        
        count_diff = site_count - sum(counts)
        counts[-1] += count_diff

        start_index = 0
        for (elem, _), count in zip(substitution_list, counts):
            for i in range(start_index, start_index + count):
                substitutions.append((elem, site_indices[i]))
            start_index += count
        return substitutions

    na_rules = substitution_rules[tm_type]["Na"]
    cl_rules = substitution_rules[tm_type]["Cl"]

    na_substitutions = _substitute_sites(na_sites, na_rules)
    cl_substitutions = _substitute_sites(cl_sites, cl_rules)

    # 3. Execute atom replacement
    for elem, site_idx in na_substitutions + cl_substitutions:
        new_structure.replace(site_idx, Element(elem))

    target_order = substitution_rules[tm_type]["order"]
    ordered_structure = reorder_atoms(new_structure, target_order)

    return ordered_structure


# ----------------------------------------------------------------------
# SQS structure generation: SMOL & ATAT
# ----------------------------------------------------------------------

def generate_sqs_structure_with_smol(
    species=None,
    supercell_matrix=None,
    cutoffs=None,
    mcmc_steps: int = 10000,
    temperatures=None,
    max_save_num=None,
    progress: bool = True,
    num_best_structures: int = 1,
):
    """
    使用 SMOL 生成 Special Quasi-random Structures (SQS)。
    """
    if species is None:
        species = [
            {"Li": 1.2 / 2.0, "Mn": 0.4 / 2.0, "Ti": 0.4 / 2.0},
            {"O": 1.0},
        ]
    if supercell_matrix is None:
        supercell_matrix = [[5, 0, 0], [0, 6, 0], [0, 0, 8]]
    if cutoffs is None:
        cutoffs = {2: 7, 3: 5}

    structure = Structure.from_spacegroup(
        "Fm-3m",
        lattice=Lattice.cubic(4.2),
        species=species,
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )

    primitive = structure.get_primitive_structure()
    supercell = primitive.make_supercell(supercell_matrix)

    sqs_generator = StochasticSQSGenerator.from_structure(
        structure=supercell,
        cutoffs=cutoffs,
        supercell_size=1,
        feature_type="correlation",
        match_weight=1.0,
    )

    sqs_generator.generate(
        mcmc_steps=mcmc_steps,
        temperatures=temperatures,
        max_save_num=max_save_num,
        progress=progress,
    )

    best_sqs = sqs_generator.get_best_sqs(
        num_structures=num_best_structures,
        remove_duplicates=True,
    )

    return best_sqs if best_sqs else None


def generate_sqs_structure_with_atat(
    species=None,
    clusters=None,
    scaling=36,
    instances: int = 1,
    search_time: int = 15,
):
    """
    Generate SQS structures using ATAT's `mcsqs` (requires local installation of ATAT).
    """
    if species is None:
        species = [
            {"Li": 1.2 / 2.0, "Mn": 0.4 / 2.0, "Ti": 0.4 / 2.0},
            {"O": 1.0},
        ]
    if clusters is None:
        clusters = {2: 7, 3: 5}

    structure = Structure.from_spacegroup(
        "Fm-3m",
        lattice=Lattice.cubic(4.2),
        species=species,
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )

    primitive = structure.get_primitive_structure()

    mcsqs_result = run_mcsqs(
        structure=primitive,
        clusters=clusters,
        scaling=scaling,
        instances=instances,
        search_time=search_time,
    )

    return mcsqs_result.bestsqs

