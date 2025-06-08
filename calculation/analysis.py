from pymatgen.analysis.local_env import CrystalNN, CutOffDictNN
from pymatgen.core.structure import Structure, Element
import numpy as np
from scipy.spatial import ConvexHull
import warnings
from collections import defaultdict
import math

def check_coordination_crystalnn(structure_file, element_symbol='Li', expected_cn=6):
    """
    Use the CrystalNN method to check if the coordination number of a specified element meets expectations
    
    Parameters:
        structure_file: Path to the structure file or a Structure object
        element_symbol: Element symbol to check (e.g. 'Li')
        expected_cn: Expected coordination number (e.g. 6)
    """

    if isinstance(structure_file, str):
        structure = Structure.from_file(structure_file)
    else:
        structure = structure_file
    
    cnn = CrystalNN()
    element = Element(element_symbol)
    element_indices = []
    
    for i, site in enumerate(structure):
        if site.species.elements[0] == element:
            element_indices.append(i)
    
    print(f"Found {len(element_indices)} {element_symbol} atoms")
    
    results = []
    for idx in element_indices:
        cn = cnn.get_cn(structure, idx)
        if cn != expected_cn:
            print(f"{element_symbol} atom #{idx} coordination number: {cn} (expected: {expected_cn})")
            results.append((idx, cn))
    
    return results

def check_coordination_cutoffnn(structure_file, element_symbol, expected_cn, 
                               cut_off_dict=None, default_cutoff=3.0):
    """
    Use the CutOffDictNN method to check if the coordination number of a specified element meets expectations
    
    Parameters:
        structure_file: Path to the structure file or a Structure object
        element_symbol: Element symbol to check (e.g. 'Li')
        expected_cn: Expected coordination number (e.g. 6)
        cut_off_dict: Coordination bond cutoff distance dictionary (e.g. {('Li', 'O'): 3.0})
        default_cutoff: Default cutoff distance to use when cut_off_dict is None
    """
    if isinstance(structure_file, str):
        structure = Structure.from_file(structure_file)
    else:
        structure = structure_file
    
    element = Element(element_symbol)
    element_indices = []
    
    for i, site in enumerate(structure):
        if site.species.elements[0] == element:
            element_indices.append(i)
    
    if cut_off_dict is None:
        cut_off_dict = {}
        unique_elements = {str(el) for site in structure for el in site.species.elements}
        for neighbor_symbol in unique_elements:
            if neighbor_symbol != element_symbol:
                cut_off_dict[(element_symbol, neighbor_symbol)] = default_cutoff
    
    cut_off_nn = CutOffDictNN(cut_off_dict)
    
    print(f"Found {len(element_indices)} {element_symbol} atoms")
    print(f"Using cutoff dictionary: {cut_off_dict}")
    
    results = []
    for idx in element_indices:
        cn = cut_off_nn.get_cn(structure, idx)
        if cn != expected_cn:
            print(f"{element_symbol} atom #{idx} coordination number: {cn} (expected: {expected_cn})")
            results.append((idx, cn))
    
    return results

def calculate_bond_lengths_crystalnn(structure_file, element_symbol):
    """
    Calculate all bond lengths for a specified element using the CrystalNN method
    
    Parameters:
        structure_file: Path to the structure file or a Structure object
        element_symbol: Element symbol to analyze (e.g. 'Li')
        
    Returns:
        all_bond_lengths: Bond length list for each target atom (list of lists)
        flattened_lengths: Flat list of all bond lengths
        average_length: Average bond length
    """
    if isinstance(structure_file, str):
        structure = Structure.from_file(structure_file)
    else:
        structure = structure_file
    
    cnn = CrystalNN()
    element = Element(element_symbol)
    element_indices = []
    
    for i, site in enumerate(structure):
        if site.species.elements[0] == element:
            element_indices.append(i)
    
    all_bond_lengths = []
    
    for idx in element_indices:
        bond_lengths = []
        num_nn = cnn.get_cn(structure, idx)
        
        for j in range(num_nn):
            neighbor_idx = cnn.get_nn_info(structure, idx)[j]['site_index']
            length = structure.get_distance(idx, neighbor_idx)
            bond_lengths.append(length)
        
        all_bond_lengths.append(bond_lengths)
    
    flattened_lengths = [length for sublist in all_bond_lengths for length in sublist]
    average_length = sum(flattened_lengths) / len(flattened_lengths) if flattened_lengths else 0
    
    print(f"Total bonds: {len(flattened_lengths)}")
    print(f"Average bond length: {average_length:.4f} Å")
    
    return all_bond_lengths, flattened_lengths, average_length

def calculate_bond_lengths_cutoffnn(structure_file, element_symbol, cut_off_dict):
    """
    Calculate all bond lengths for a specified element using the CutOffDictNN method
    
    Parameters:
        structure_file: Path to the structure file or a Structure object
        element_symbol: Element symbol to analyze (e.g. 'Li')
        cut_off_dict: Coordination bond cutoff distance dictionary (e.g. {('Li', 'O'): 3.0})
        
    Returns:
        all_bond_lengths: Bond length list for each target atom (list of lists)
        flattened_lengths: Flat list of all bond lengths
        average_length: Average bond length
    """
    if isinstance(structure_file, str):
        structure = Structure.from_file(structure_file)
    else:
        structure = structure_file
    
    cut_off_nn = CutOffDictNN(cut_off_dict)
    element = Element(element_symbol)
    element_indices = []
    
    for i, site in enumerate(structure):
        if site.species.elements[0] == element:
            element_indices.append(i)
    
    print(f"Found {len(element_indices)} {element_symbol} atoms")
    print(f"Using cutoff dictionary: {cut_off_dict}")
    
    all_bond_lengths = []
    
    for idx in element_indices:
        bond_lengths = []
        num_nn = cut_off_nn.get_cn(structure, idx)
        
        for j in range(num_nn):
            neighbor_idx = cut_off_nn.get_nn_info(structure, idx)[j]['site_index']
            length = structure.get_distance(idx, neighbor_idx)
            bond_lengths.append(length)
        
        all_bond_lengths.append(bond_lengths)
    
    flattened_lengths = [length for sublist in all_bond_lengths for length in sublist]
    average_length = sum(flattened_lengths) / len(flattened_lengths) if flattened_lengths else 0
    
    print(f"Total bonds: {len(flattened_lengths)}")
    print(f"Average bond length: {average_length:.4f} Å")
    
    return all_bond_lengths, flattened_lengths, average_length

# Ignore convex hull calculation warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def analyze_polyhedra(structure, center_species, ligand_species):
    """
    Analyze polyhedra in a structure with specified central atoms:
    1. Output volume information for each calculable polyhedron
    2. Output statistical information about different coordination polyhedra
    
    Parameters:
        structure: pymatgen.Structure
        center_species: Central atom element (e.g. 'Li')
        ligand_species: Ligand atom element (e.g. 'O')
    
    Returns:
        tuple: (Volume list, coordination statistics dictionary, 
                successful polyhedra count, total polyhedra count)
    """
    cnn = CrystalNN(distance_cutoffs=None, cation_anion=False, weighted_cn=False)
    coordination_stats = defaultdict(int)  # For counting different coordination numbers
    
    print(f"Analyzing {center_species}-{ligand_species} polyhedra...\n")
    
    total_polyhedra = 0  # Total polyhedra detected
    calculated_polyhedra = 0  # Successfully calculated polyhedra
    polyhedra_counter = 0  # Polyhedron numbering counter
    
    poly_volume_list = []  # Store volume data
    
    for i, site in enumerate(structure):
        if site.species_string != center_species:
            continue
            
        total_polyhedra += 1
        neighbors = cnn.get_nn_info(structure, i)
        
        ligand_neighbors = [n for n in neighbors if n['site'].species_string == ligand_species]
        coordination = len(ligand_neighbors)
        coordination_stats[coordination] += 1
        
        ligand_coords = [n['site'].coords for n in ligand_neighbors]
        poly_volume = None
        
        if coordination >= 4:
            try:
                hull = ConvexHull(ligand_coords)
                poly_volume = hull.volume
                calculated_polyhedra += 1
                
                polyhedra_counter += 1
            except Exception as e:
                polyhedra_counter += 1
        
        poly_volume_list.append(poly_volume)
    
    print("\nPolyhedral coordination statistics:")
    sorted_coordinations = sorted(coordination_stats.keys())
    for coord in sorted_coordinations:
        count = coordination_stats[coord]
        print(f"  {center_species}{ligand_species}{coord} count: {count}")
    
    return poly_volume_list, coordination_stats, calculated_polyhedra, total_polyhedra

def analyze_polyhedra_cutoffnn(structure, center_species, ligand_species, cutoff_distance=3.0):
    """
    Analyze polyhedra in a structure with specified central atoms:
    1. Output statistical information about different coordination polyhedra
    2. Return volume information for central atoms (values where calculable, None otherwise)
    
    Parameters:
        structure: pymatgen.Structure
        center_species: Central atom element (e.g. 'Li')
        ligand_species: Ligand atom element (e.g. 'O')
        cutoff_distance: Max center-ligand atom distance for CutOffDictNN
        
    Returns:
        list: Polyhedron volume for each central atom (value if calculable, None otherwise)
    """
    cut_off_dict = {(center_species, ligand_species): cutoff_distance}
    cut_off_nn = CutOffDictNN(cut_off_dict)
    coordination_stats = defaultdict(int)
    
    print(f"Analyzing {center_species}-{ligand_species} polyhedra using cutoff distance {cutoff_distance:.2f}Å...\n")
    
    poly_volume_list = []
    
    for i, site in enumerate(structure):
        if site.species_string != center_species:
            continue
            
        neighbors = cut_off_nn.get_nn_info(structure, i)
        ligand_neighbors = [n for n in neighbors if n['site'].species_string == ligand_species]
        coordination = len(ligand_neighbors)
        coordination_stats[coordination] += 1
        
        ligand_coords = [n['site'].coords for n in ligand_neighbors]
        poly_volume = None
        
        if coordination >= 4:
            try:
                hull = ConvexHull(ligand_coords)
                poly_volume = hull.volume
            except Exception:
                pass
        
        poly_volume_list.append(poly_volume)
    
    print("\nPolyhedral coordination statistics:")
    sorted_coordinations = sorted(coordination_stats.keys())
    for coord in sorted_coordinations:
        count = coordination_stats[coord]
        print(f"  {center_species}{ligand_species}{coord} count: {count}")
    
    return poly_volume_list

def calculate_polyhedral_distortion(structure_file, element_symbol):
    """
    Calculate distortion factors for polyhedra formed by a specified element
    
    Parameters:
        structure_file: Path to the structure file or a Structure object
        element_symbol: Element symbol to analyze (e.g. 'Li')
        
    Returns:
        distortion_factors: List of distortion factors for each polyhedron
        avg_distortion: Average distortion factor for all polyhedra
    """
    if isinstance(structure_file, str):
        structure = Structure.from_file(structure_file)
    else:
        structure = structure_file
    
    cnn = CrystalNN()
    element = Element(element_symbol)
    element_indices = []
    
    for i, site in enumerate(structure):
        if site.species.elements[0] == element:
            element_indices.append(i)
    
    print(f"Found {len(element_indices)} {element_symbol} atoms")
    
    distortion_factors = []
    
    for idx in element_indices:
        bond_lengths = []
        num_nn = cnn.get_cn(structure, idx)
        
        for j in range(num_nn):
            neighbor_idx = cnn.get_nn_info(structure, idx)[j]['site_index']
            length = structure.get_distance(idx, neighbor_idx)
            bond_lengths.append(length)
        
        avg_bond = np.mean(bond_lengths)
        total_deviation = 0.0
        for length in bond_lengths:
            total_deviation += math.fabs((length - avg_bond) / avg_bond)
        
        distortion = total_deviation / num_nn
        distortion_factors.append(distortion)
    
    avg_distortion = np.mean(distortion_factors) if distortion_factors else 0.0
    
    print(f"{element_symbol} polyhedra distortion factors:", [f"{d:.4f}" for d in distortion_factors])
    print(f"Average distortion factor for {element_symbol} polyhedra: {avg_distortion:.4f}")
    
    return distortion_factors, avg_distortion

def calculate_polyhedral_distortion_cutoffnn(structure, element_symbol, cut_off_dict):
    """
    Calculate distortion factors for polyhedra using CutOffDictNN
    
    Parameters:
        structure: Path to structure file or Structure object
        element_symbol: Element symbol to analyze (e.g. 'Li')
        cut_off_dict: Cutoff distance dictionary e.g. {('Li','O'): 3.0}
        
    Returns:
        distortion_factors: List of distortion factors for each polyhedron
        avg_distortion: Average distortion factor
        coordination_numbers: List of coordination numbers for each polyhedron
    """
    if isinstance(structure, str):
        structure = Structure.from_file(structure)
    
    cut_off_nn = CutOffDictNN(cut_off_dict)
    element = Element(element_symbol)
    element_indices = []
    
    for i, site in enumerate(structure):
        if site.species.elements[0] == element:
            element_indices.append(i)
    
    print(f"Found {len(element_indices)} {element_symbol} atoms")
    
    distortion_factors = []
    coordination_numbers = []
    valid_count = 0  # Atoms with valid polyhedra
    
    for idx in element_indices:
        neighbors = cut_off_nn.get_nn_info(structure, idx)
        if len(neighbors) == 0:
            continue
            
        bond_lengths = []
        for neighbor in neighbors:
            bond_lengths.append(neighbor['weight'])
        
        avg_bond = np.mean(bond_lengths)
        coordination = len(bond_lengths)
        coordination_numbers.append(coordination)
        
        total_deviation = 0.0
        for length in bond_lengths:
            total_deviation += math.fabs((length - avg_bond) / avg_bond)
        
        distortion = total_deviation / coordination
        distortion_factors.append(distortion)
        valid_count += 1
    
    avg_distortion = np.mean(distortion_factors) if distortion_factors else 0.0
    
    print(f"\nSummary:")
    print(f"Valid {element_symbol} polyhedra count: {valid_count}")
    print(f"Distortion factor range: {min(distortion_factors):.4f} - {max(distortion_factors):.4f}")
    print(f"Average distortion factor: {avg_distortion:.4f}")
    
    return distortion_factors, avg_distortion, coordination_numbers