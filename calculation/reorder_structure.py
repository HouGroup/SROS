from pymatgen.core import Structure
from typing import Iterable, List


def reorder_atoms(structure: Structure, input_order: List[str]) -> Structure:
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


def reorder_atoms_flexible(structure: Structure, input_order: Iterable[str]) -> Structure:
    """
    Reorder atoms by the requested element sequence while preserving any
    elements that are not listed in ``input_order``.

    This is useful for workflows where oxidation states, model outputs, or
    extra dopants introduce species beyond the main composition list.
    """
    requested_order = list(input_order)
    remaining_order = sorted(
        {site.specie.symbol for site in structure} - set(requested_order)
    )
    return reorder_atoms(structure, requested_order + remaining_order)
