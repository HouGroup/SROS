import numpy as np
from pymatgen.core import Lattice, Structure, Element
from collections import defaultdict
import random
from smol.capp.generate.special.sqs import StochasticSQSGenerator
from pymatgen.command_line.mcsqs_caller import run_mcsqs

from .reorder_structure import reorder_atoms_flexible

def create_original_structure():
    """
    Create a prototype structure based on input parameters.
    """
    # Lattice vectors (3x3 matrix, units: Å)
    lattice = [
        [0.0, 2.1, 2.1],
        [2.1, 0.0, 2.1],
        [2.1, 2.1, 0.0]
    ]
    
    # Atomic species and coordinates (fractional coordinates)
    species = ["Na", "Cl"]  # Element sequence
    coords = [
        [0.0, 0.0, 0.0],    # Na atom coordinate
        [0.5, 0.5, 0.5]     # Cl atom coordinate
    ]
    
    # Create pymatgen Structure object
    structure = Structure(lattice, species, coords)
    return structure

def make_supercell(scaling_matrix=[[-1, 1, 1], [1, -1, 1], [1, 1, -1]]):
    """
    Create a supercell from the prototype structure using an integer transformation matrix.
    
    :param scaling_matrix: 3x3 integer matrix defining the supercell transformation
    :return: Supercell Structure object
    """
    # Create prototype structure
    original = create_original_structure()
    
    # Ensure matrix is integer type
    scaling_matrix = np.array(scaling_matrix, dtype=int)
    
    # Validate matrix dimensions
    if scaling_matrix.shape != (3, 3):
        raise ValueError("Scaling matrix must be a 3x3 integer matrix")
    
    # Create supercell
    supercell = original.copy()
    supercell.make_supercell(scaling_matrix)
    
    return supercell

def modify_structure(li_content, mn_content, input_structure):
    """
    Modify a structure according to given Li and Mn concentrations by randomly substituting atoms.
    
    Parameters:
    - li_content: float, Li concentration
    - mn_content: float, Mn concentration
    - input_structure: Input Structure object to modify

    Returns:
    - Structure: Modified structure with atom substitutions
    """
    structure = input_structure
    
    # Convert Na sites to Li
    na_sites = [i for i, site in enumerate(structure) 
                if site.species_string == "Na"]
    for i in na_sites:
        structure[i] = Element("Li")

    # Convert Cl sites to O
    cl_sites = [i for i, site in enumerate(structure) 
                if site.species_string == "Cl"]
    for i in cl_sites:
        structure[i] = Element("O")
        
    # Calculate Ti content
    ti_content = 2 - li_content - mn_content

    # Find Li sites
    li_sites = [i for i, site in enumerate(structure) 
                if site.species_string == "Li"]

    # Randomly convert Li to Mn
    num_mn = int(round(len(li_sites) * mn_content / 2))
    selected_mn = random.sample(li_sites, num_mn)
    for i in selected_mn:
        structure[i] = Element("Mn")

    # Convert remaining Li to Ti
    remaining_li = [num for num in li_sites if num not in selected_mn]
    num_ti = int(round(len(li_sites) * ti_content / 2))
    selected_ti = random.sample(remaining_li, num_ti)
    for i in selected_ti:
        structure[i] = Element("Ti")

    # Reorder atoms in specified sequence
    ordered_structure = reorder_atoms_flexible(structure, ["Li", "Mn", "Ti", "O"])
    
    return ordered_structure

def substitute_elements(structure: Structure, tm_type: str, seed: int = None) -> Structure:
    """
    Perform element substitution in a Na-Cl structure while maintaining concentration ratios.
    
    Args:
        structure: Original structure containing Na and Cl atoms
        tm_type: Substitution rule type ("TM2", "TM4", "TM6")
        seed: Random seed (optional, for reproducibility)

    Returns:
        Structure: Modified structure after substitution
    """
    # Define substitution rules
    substitution_rules = {
        "TM2": {
            "Na": [("Li", 0.65), ("Mn", 0.20), ("Ti", 0.15)],
            "Cl": [("O", 0.85), ("F", 0.15)]
        },
        "TM4": {
            "Na": [("Li", 0.65), ("Mn", 0.20), ("Ti", 0.05), ("Nb", 0.10)],
            "Cl": [("O", 0.85), ("F", 0.15)]
        },
        "TM6": {
            "Na": [("Li", 0.65), ("Mn", 0.10), ("Co", 0.05), ("Cr", 0.05),
                   ("Ti", 0.05), ("Nb", 0.10)],
            "Cl": [("O", 0.85), ("F", 0.15)]
        }
    }

    # Validate input
    if tm_type not in substitution_rules:
        raise ValueError(f"Unsupported tm_type: {tm_type}. Use 'TM2', 'TM4', or 'TM6'.")

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Create copy for modification
    new_structure = structure.copy()

    # Collect Na and Cl site indices
    na_sites = [i for i, site in enumerate(new_structure) 
                if site.specie == Element.Na]
    cl_sites = [i for i, site in enumerate(new_structure) 
                if site.specie == Element.Cl]

    # Helper function for proportional substitution
    def substitute_sites(site_indices, substitution_list):
        np.random.shuffle(site_indices)  # Randomize order
        site_count = len(site_indices)
        substitutions = []
        
        # Calculate substitution counts
        counts = []
        for elem, fraction in substitution_list:
            count = int(round(site_count * fraction))
            counts.append(count)
        
        # Adjust counts to match total site count
        count_diff = site_count - sum(counts)
        if count_diff != 0:
            counts[-1] += count_diff  # Compensate in final element
        
        # Assign substitutions
        start_index = 0
        for (elem, _), count in zip(substitution_list, counts):
            substitutions.extend([(elem, idx) 
                                for idx in site_indices[start_index:start_index+count]])
            start_index += count
        return substitutions

    # Substitute Na sites
    na_rules = substitution_rules[tm_type]["Na"]
    na_substitutions = substitute_sites(na_sites, na_rules)

    # Substitute Cl sites
    cl_rules = substitution_rules[tm_type]["Cl"]
    cl_substitutions = substitute_sites(cl_sites, cl_rules)

    # Apply all substitutions
    for elem, site_idx in na_substitutions + cl_substitutions:
        new_structure.replace(site_idx, Element(elem))

    return new_structure

def generate_sqs_structure_with_smol(
    species=[{"Li": 1.2/2.0, "Mn": 0.4/2.0, "Ti": 0.4/2.0}, {"O": 1.0}],
    supercell_matrix=[[5, 0, 0], [0, 6, 0], [0, 0, 8]],
    cutoffs={2: 7, 3: 5},
    mcmc_steps=10000,
    temperatures=None,
    max_save_num=None,
    progress=True,
    num_best_structures=1
):
    """
    Generate optimized Special Quasi-random Structures (SQS) using SMOL.
    
    Parameters:
    species (list): Species concentrations for each site
    supercell_matrix (list): Supercell transformation matrix
    cutoffs (dict): Cluster cutoffs {order: radius}
    mcmc_steps (int): MCMC steps, default 10000
    temperatures (list): Annealing temperature schedule
    max_save_num (int): Maximum structures to save
    progress (bool): Show progress bar
    num_best_structures (int): Number of best structures to return
    
    Returns:
    Structure: Optimized SQS structure
    """
    # Create prototype structure
    structure = Structure.from_spacegroup(
        "Fm-3m",
        lattice=Lattice.cubic(4.2),
        species=species,
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    
    # Get primitive structure
    primitive = structure.get_primitive_structure()
    
    # Create supercell
    supercell = primitive.make_supercell(supercell_matrix)
    
    # Initialize SQS generator
    sqs_generator = StochasticSQSGenerator.from_structure(
        structure=supercell,
        cutoffs=cutoffs,
        supercell_size=1,
        feature_type="correlation",
        match_weight=1.0
    )
    
    # Generate SQS structures
    sqs_generator.generate(
        mcmc_steps=mcmc_steps,
        temperatures=temperatures,
        max_save_num=max_save_num,
        progress=progress
    )
    
    # Retrieve best SQS structures
    best_sqs = sqs_generator.get_best_sqs(
        num_structures=num_best_structures,
        remove_duplicates=True
    )
    
    return best_sqs if best_sqs else None


def generate_sqs_structure_with_atat(
    species=[{"Li": 1.2/2.0, "Mn": 0.4/2.0, "Ti": 0.4/2.0}, {"O": 1.0}],
    clusters={2: 7, 3: 5},
    scaling=36,
    instances=1,
    search_time=15,
):
    """
    Generate Special Quasi-random Structures (SQS) using ATAT mcsqs algorithm.
    Requires ATAT to be installed.
    
    Parameters:
    species (list): Species concentrations for each site
    clusters (dict): Cluster definitions {size: cutoff}
    scaling (int|list): Supercell scaling factor
    instances (int): Parallel instances
    search_time (int): Search time in minutes
    
    Returns:
    Structure: Optimized SQS structure
    """
    # Create prototype structure
    structure = Structure.from_spacegroup(
        "Fm-3m",
        lattice=Lattice.cubic(4.2),
        species=species,
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    
    # Get primitive structure
    primitive = structure.get_primitive_structure()

    # Run mcsqs algorithm
    mcsqs_result = run_mcsqs(
        structure=primitive,
        clusters=clusters,
        scaling=scaling,
        instances=instances,
        search_time=search_time
    )
    
    # Return best SQS structure
    return mcsqs_result.bestsqs