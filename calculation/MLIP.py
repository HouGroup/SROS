import json
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from sevenn.sevennet_calculator import SevenNetCalculator
from ase.io import read
import torch
from tqdm import tqdm
from pymatgen.core.structure import Structure
import random
from ase.optimize import BFGS  # For structure optimization

def initialize_calculator(model_path, device='cpu'):
    """
    Initialize the SevenNet calculator
    
    Args:
        model_path: Path to the model checkpoint file
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        SevenNetCalculator instance
    """
    sevennet_calc = SevenNetCalculator(model_paths=[model_path], device=device)
    return sevennet_calc

def relax_structure(input_file, calculator):
    """
    Perform structure relaxation using the SevenNet calculator
    
    Args:
        input_file: Path to input structure file
        calculator: Initialized SevenNet calculator
        
    Returns:
        tuple: (relaxed Structure object, final potential energy)
    """
    # Read input structure
    structure = Structure.from_file(input_file)
    atoms = AseAtomsAdaptor.get_atoms(structure)
    
    # Set calculator
    atoms.calc = calculator
    
    # Perform structural relaxation
    optimizer = BFGS(atoms)  # Uses BFGS algorithm for optimization
    optimizer.run(fmax=0.05, steps=500)  # Convergence criteria: max force < 0.05 eV/Å
    
    # Get results
    energy = atoms.get_potential_energy()
    final_structure = Structure.from_ase_atoms(atoms)
    
    return final_structure