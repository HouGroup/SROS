from pymatgen.core import Structure, Element, Site, Species
from smol.io import load_work
import random
import numpy as np
from smol.moca import Ensemble
from smol.moca import Sampler
import os
import shutil
import re

from .reorder_structure import reorder_atoms_flexible

def calculate_mn_ratios(li_content, mn_content):
    """Calculate the valence state ratios of Mn based on Li and Mn contents."""
    # Calculate Ti content
    ti_content = 2 - li_content - mn_content
    
    # Calculate total positive charge
    positive_charge = 1 * li_content + 4 * ti_content + 3 * mn_content
    
    # Initialize Mn valence state ratios
    mn2_ratio, mn3_ratio, mn4_ratio = 0, 0, 0
    
    # Determine Mn ratios based on charge balance
    if positive_charge == 4:
        mn2_ratio = 0
        mn3_ratio = mn_content
        mn4_ratio = 0
    elif positive_charge > 4:
        # Mn contains both +2 and +3 valence states
        mn2_ratio = min(positive_charge - 4, mn_content)
        mn3_ratio = mn_content - mn2_ratio
        mn4_ratio = 0
    elif positive_charge < 4:
        # Mn contains both +3 and +4 valence states
        mn4_ratio = min(4 - positive_charge, mn_content)
        mn3_ratio = mn_content - mn4_ratio
        mn2_ratio = 0
    
    # Round to 4 decimal places
    mn2_ratio = round(mn2_ratio, 4)
    mn3_ratio = round(mn3_ratio, 4)
    mn4_ratio = round(mn4_ratio, 4)
    
    return mn2_ratio, mn3_ratio, mn4_ratio

def assign_element_numbers(structure, mn2_ratio, mn3_ratio, mn4_ratio):
    """Assign numerical identifiers to elements based on valence states."""
    # Define mapping from element symbols to numerical identifiers
    element_to_number = {'Li+1': 0, 'Mn+2': 2, 'Mn+3': 3, 'Mn+4': 4, 'Ti+4': 1, 'O-2': 0}
    
    # Initialize occupancy array
    occupancies = np.zeros(structure.num_sites, dtype=int)
    
    # Find all Mn atom indices
    mn_indices = [i for i, site in enumerate(structure) if site.specie.symbol == 'Mn']
    
    # Calculate counts for each Mn valence state
    total_mn = len(mn_indices)
    
    if (mn2_ratio + mn3_ratio + mn4_ratio) == 0:
        num_mn2 = num_mn3 = num_mn4 = 0
    else:
        num_mn2 = int(total_mn * mn2_ratio / (mn2_ratio + mn3_ratio + mn4_ratio))
        num_mn3 = int(total_mn * mn3_ratio / (mn2_ratio + mn3_ratio + mn4_ratio))
        num_mn4 = total_mn - num_mn2 - num_mn3  # Remaining assigned to Mn+4
    
    # Randomly assign valence states to Mn atoms
    mn_2_indices = random.sample(mn_indices, num_mn2)
    remaining_indices = list(set(mn_indices) - set(mn_2_indices))
    mn_3_indices = random.sample(remaining_indices, num_mn3)
    mn_4_indices = list(set(remaining_indices) - set(mn_3_indices))
    
    # Assign numerical identifiers to all atoms
    for i, site in enumerate(structure):
        symbol = site.specie.symbol
        
        if symbol == 'Li':
            occupancies[i] = element_to_number['Li+1']  # Li assigned +1
        elif symbol == 'Mn':
            # Assign based on valence state
            if i in mn_2_indices:
                occupancies[i] = element_to_number['Mn+2']  # Assigned as Mn+2
            elif i in mn_3_indices:
                occupancies[i] = element_to_number['Mn+3']  # Assigned as Mn+3
            elif i in mn_4_indices:
                occupancies[i] = element_to_number['Mn+4']  # Assigned as Mn+4
        elif symbol == 'Ti':
            occupancies[i] = element_to_number['Ti+4']  # Ti assigned +4
        elif symbol == 'O':
            occupancies[i] = element_to_number['O-2']  # O assigned -2
    
    return np.array(occupancies)

def run_cemc(li_content, 
             mn_content,
             input_structure,
             ce_model_path,
             sc_matrix=np.array([[5, 0, 0], [0, 6, 0], [0, 0, 8]]),  # Supercell matrix
             temperature=1273,
             mc_step=120000
            ):
    """
    Run a Cluster Expansion Monte Carlo (CEMC) simulation.
    
    Parameters:
        li_content (float): Li composition fraction
        mn_content (float): Mn composition fraction
        input_structure (Structure): Initial structure
        ce_model_path (str): Path to Cluster Expansion model
        sc_matrix (ndarray): Supercell transformation matrix
        temperature (float): Simulation temperature in Kelvin
        mc_step (int): Number of Monte Carlo steps
        
    Returns:
        Structure: Final structure after CEMC simulation
    """
    # Load Cluster Expansion model
    work = load_work(ce_model_path)
    expansion = work['ClusterExpansion']
    
    # Create ensemble from Cluster Expansion
    ensemble = Ensemble.from_cluster_expansion(expansion, sc_matrix)
    
    # Create sampler
    sampler = Sampler.from_ensemble(ensemble, temperature)
    
    print(f"Processing li_content={li_content}, mn_content={mn_content}")

    # Calculate Mn valence state ratios
    mn2_ratio, mn3_ratio, mn4_ratio = calculate_mn_ratios(
        li_content=li_content, mn_content=mn_content
    )
    
    # Initialize occupancies based on composition
    init_occu = assign_element_numbers(input_structure, mn2_ratio, mn3_ratio, mn4_ratio)
    
    # Run Monte Carlo simulation
    sampler.run(
        mc_step,
        initial_occupancies=init_occu,
        thin_by=1000,  # Save every 1000th sample
        progress=True
    )
    
    # Get final structure
    samples = sampler.samples
    occupancy = samples.get_occupancies()[-1]  # Last sample (every 100,000 steps)
    structure = ensemble.processor.structure_from_occupancy(occupancy)

    # Reorder atoms in the structure
    ordered_structure = reorder_atoms_flexible(structure, ["Li", "Mn", "Ti", "O"])
    
    return ordered_structure