from pymatgen.core import Structure
from typing import List

# Configuration parameters - atom ordering by type
tm2_order = ["Li", "Mn", "Ti", "O", "F"]
tm4_order = ["Li", "Mn", "Ti", "Nb", "O", "F"]
tm6_order = ["Li", "Mn", "Co", "Cr", "Ti", "Nb", "O", "F"]

def reorder_atoms(structure: Structure, tm_type: str) -> Structure:
    """
    Reorder atoms in structure according to specified sequence for transition metal type
    
    Args:
        structure: Input structure (should contain Li, Mn, Ti, O, F etc.)
        tm_type: Transition metal substitution type ("TM2", "TM4", "TM6")

    Returns:
        Structure: New structure with atoms ordered by the specified sequence
    """
    # Select ordering sequence based on tm_type
    order: List[str]
    if tm_type == "TM2":
        order = tm2_order
    elif tm_type == "TM4":
        order = tm4_order
    elif tm_type == "TM6":
        order = tm6_order
    else:
        valid_types = ", ".join(["TM2", "TM4", "TM6"])
        raise ValueError(f"Unsupported tm_type: '{tm_type}'. Valid types: {valid_types}")

    # Collect atoms in specified order (preserving coordinates and properties)
    ordered_sites = []
    for element in order:
        for site in structure:
            if site.specie.symbol == element:
                ordered_sites.append(site)
    
    # Validate all atoms were processed
    if len(ordered_sites) != len(structure):
        missing = len(structure) - len(ordered_sites)
        error_message = (
            f"{missing} atoms could not be matched to the order list. "
            f"Check if input structure matches tm_type {tm_type}."
        )
        raise RuntimeError(error_message)

    # Process site properties (convert list to dictionary)
    site_properties = {}
    if ordered_sites and hasattr(ordered_sites[0], 'properties'):
        # Get all property keys
        keys = ordered_sites[0].properties.keys()
        for key in keys:
            site_properties[key] = [site.properties.get(key) for site in ordered_sites]

    # Create new structure (preserving lattice parameters)
    return Structure(
        lattice=structure.lattice,
        species=[site.species for site in ordered_sites],
        coords=[site.frac_coords for site in ordered_sites],
        coords_are_cartesian=False,
        site_properties=site_properties
    )

def reorder_atoms_flexible(structure: Structure, input_order: List[str]) -> Structure:
    """
    Reorder atoms in structure according to a custom atom sequence
    
    Args:
        structure: Input structure (should contain specified elements)
        input_order: List of element symbols for desired order

    Returns:
        Structure: New structure with atoms ordered as specified
    """
    # Collect atoms in specified order (preserving coordinates and properties)
    ordered_sites = []
    for element in input_order:
        for site in structure:
            if site.specie.symbol == element:
                ordered_sites.append(site)
    
    # Validate all atoms were processed
    if len(ordered_sites) != len(structure):
        missing = len(structure) - len(ordered_sites)
        missing_elements = set(site.specie.symbol for site in structure) - set(input_order)
        error_message = (
            f"{missing} atoms ({', '.join(missing_elements)}) not in order list. "
            f"Check if input structure matches the specified order."
        )
        raise RuntimeError(error_message)

    # Process site properties (convert list to dictionary)
    site_properties = {}
    if ordered_sites and hasattr(ordered_sites[0], 'properties'):
        # Get all property keys
        keys = ordered_sites[0].properties.keys()
        for key in keys:
            site_properties[key] = [site.properties.get(key) for site in ordered_sites]

    # Create new structure (preserving lattice parameters)
    return Structure(
        lattice=structure.lattice,
        species=[site.species for site in ordered_sites],
        coords=[site.frac_coords for site in ordered_sites],
        coords_are_cartesian=False,
        site_properties=site_properties
    )