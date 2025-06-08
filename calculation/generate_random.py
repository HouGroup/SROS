import numpy as np
from pymatgen.core import Structure, Element
from collections import defaultdict

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
    
    # Atomic species and fractional coordinates
    species = ["Na", "Cl"]  # Species sequence
    coords = [
        [0.0, 0.0, 0.0],    # Na atom coordinates
        [0.5, 0.5, 0.5]     # Cl atom coordinates
    ]
    
    # Create pymatgen Structure object
    structure = Structure(lattice, species, coords)
    return structure

def make_supercell(scaling_factors=(2, 4, 5)):
    """
    Create a supercell by scaling the prototype structure along lattice vectors.
    
    Args:
        scaling_factors: Tuple of scaling factors for (a, b, c) directions
        
    Returns:
        Structure: Supercell structure
    """
    # Create prototype structure
    original = create_original_structure()
    
    # Generate diagonal scaling matrix (e.g., (2,4,5) → [[2,0,0], [0,4,0], [0,0,5]])
    scaling_matrix = np.diag(scaling_factors)
    
    # Create supercell
    supercell = original.copy()
    supercell.make_supercell(scaling_matrix)
    
    return supercell

def substitute_elements(structure: Structure, tm_type: str, seed: int = None) -> Structure:
    """
    Perform element substitution in a Na-Cl structure while maintaining exact concentration ratios.
    
    Args:
        structure: Original structure containing Na and Cl sites
        tm_type: Substitution scheme type ("TM2", "TM4", "TM6")
        seed: Random seed for reproducibility (optional)
        
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

    # Validate input parameters
    if tm_type not in substitution_rules:
        valid_types = ", ".join(substitution_rules.keys())
        raise ValueError(f"Invalid tm_type: '{tm_type}'. Must be one of: {valid_types}")

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Create working copy of structure
    modified_structure = structure.copy()

    # Collect site indices
    na_sites = [i for i, site in enumerate(modified_structure) 
                if site.species_string == "Na"]
    cl_sites = [i for i, site in enumerate(modified_structure) 
                if site.species_string == "Cl"]

    # Helper function for proportional substitution
    def substitute_atoms(site_indices, substitution_spec):
        """
        Perform proportional substitution on atomic sites
        
        Args:
            site_indices: List of atomic indices to substitute
            substitution_spec: List of (element, fraction) tuples
            
        Returns:
            List of (element, index) substitutions to apply
        """
        np.random.shuffle(site_indices)  # Randomize selection order
        site_count = len(site_indices)
        substitutions = []
        
        # Calculate proportional counts for each element
        counts = []
        for element_symbol, concentration in substitution_spec:
            count = int(round(site_count * concentration))
            counts.append(count)
        
        # Adjust for rounding errors
        total_count = sum(counts)
        count_discrepancy = site_count - total_count
        if count_discrepancy != 0:
            counts[-1] += count_discrepancy  # Adjust last element
        
        # Assign substitutions
        start_idx = 0
        for (element_symbol, _), count in zip(substitution_spec, counts):
            end_idx = start_idx + count
            substitutions.extend([
                (element_symbol, idx) 
                for idx in site_indices[start_idx:end_idx]
            ])
            start_idx = end_idx
        return substitutions

    # Perform Na site substitutions
    na_substitutions = substitute_atoms(
        na_sites, 
        substitution_rules[tm_type]["Na"]
    )
    
    # Perform Cl site substitutions
    cl_substitutions = substitute_atoms(
        cl_sites, 
        substitution_rules[tm_type]["Cl"]
    )

    # Apply all substitutions
    for element_symbol, site_index in na_substitutions + cl_substitutions:
        modified_structure.replace(site_index, Element(element_symbol))

    return modified_structure