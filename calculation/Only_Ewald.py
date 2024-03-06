"""
Only_Ewald.py

This module provides functionality for performing an Ewald annealing process by exchanging transition metal cations in a crystal structure to minimize the system's Coulomb electrostatic energy and achieve overall electrical neutrality.

The Ewald annealing process aims to optimize the arrangement of transition metal cations within the crystal lattice to minimize the total Coulombic energy of the system. The ultimate goal is to achieve a state of electrical neutrality, ensuring a balanced distribution of charges.

Functions:
    - perform_ewald_annealing(structure): Executes the Ewald annealing process by swapping transition metal cations in the given crystal structure.
    - additional_function(): Any additional helper function relevant to the Ewald annealing process.

Example Usage:

    from mypackage import Only_Ewald
    input_structure = "D:\\Users\\ASUS\\Desktop\\Computational Practice\\0103\\Hackathon\\TM6_1_POSCAR"
    out_structure = "D:\\Users\\ASUS\\Desktop\\Computational Practice\\0103\\Hackathon\\TM6_1_POSCAR_ewald"

    Only_Ewald.Ewald(structure_path=input_structure, out_path=out_structure, DRX_choice="tm6")

Note:
    - This module assumes that the crystal structure is provided in a suitable format compatible with the Ewald annealing algorithm.
    - The effectiveness of the annealing process may depend on various factors, and the results should be validated.
    - Additional documentation and references related to the Ewald annealing method should be consulted for a deeper understanding.

Author: Liaojh
Date: January 23, 2024
"""


import copy
import numpy as np
from monty.serialization import loadfn, dumpfn
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.core import Structure
from smol.cofe import ClusterSubspace
from smol.cofe.extern import EwaldTerm
from smol.moca import EwaldProcessor
from smol.moca import Ensemble
from smol.moca import Sampler
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation

class Ewald:
    def __init__(
            self,
            structure_path: str,
            out_path: str,
            DRX_choice: str = 'tm2',  # 默认选择 tm2 函数
    ):
        input_str = Structure.from_file(structure_path)
        dumpfn(input_str, 'input_file.json')

        # 根据 DRX_choice 参数选择调用不同的函数
        if DRX_choice == 'tm2':
            self.tm2_json_file('input_file.json', 'input_file1.json')
        elif DRX_choice == 'tm4':
            self.tm4_json_file('input_file.json', 'input_file1.json')
        elif DRX_choice == 'tm6':
            self.tm6_json_file('input_file.json', 'input_file1.json')
        else:
            raise ValueError("Invalid function_choice. Please choose 'tm2', 'tm4', or 'tm6'.")

        input_str = loadfn('input_file1.json')

        empty_cutoff = {} # Defining the cut-offs as an empty dictionary will generate a subspace with only the empty cluster
        subspace = ClusterSubspace.from_cutoffs(
            structure=input_str, cutoffs=empty_cutoff, supercell_size='O2-'
        )
        subspace.add_external_term(EwaldTerm(eta=None)) # Add the external Ewald Term

        # The supercell with which we will run MC on
        sc_matrix = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        # Specifying the dielectric constant, the inverse of which is parametrized when fitting a CE with electrostatics (Example 1-1).
        dielectric = 5.0
        # Creating the Ewald Processor
        ewald_proc = EwaldProcessor(
            cluster_subspace=subspace,
            supercell_matrix=sc_matrix,
            ewald_term=EwaldTerm(),
            coefficient=1/dielectric
        )

        # Create the canonical ensemble directly from the Ewald Processor, without creating a Cluster Expansion.
        ensemble = Ensemble(processor=ewald_proc)
        # If the goal is to enumerate new structures for DFT calculations, it may be wise to limit the size of
        # your supercell such that a relaxation calculation is feasible.
        # The thermodynamics may not be the most realistic, but you can generate training structures
        # that have relatively low electrostatic energies, which may translate to lower DFT energies.
        # print(f'The supercell size for the processor is {ensemble.processor.size} prims.')
        # print(f'The ensemble has a total of {ensemble.num_sites} sites.')
        # print(f'The active sublattices are:')
        # for sublattice in ensemble.sublattices:
        #     print(sublattice)

        sampler = Sampler.from_ensemble(ensemble, temperature=2000)
        # print(f"Sampling information: {sampler.samples.metadata}")

        # Here we will just use the order disordered transformation from
        # pymatgen to get an ordered version of a prim supercell.
        # The structure will have the same composition set in the prim.
        transformation = OrderDisorderedStructureTransformation(algo=2)

        supercell = input_str.copy()
        supercell.make_supercell(sc_matrix)

        test_struct = transformation.apply_transformation(supercell)
        # print(test_struct.composition)
        init_occu = ensemble.processor.occupancy_from_structure(test_struct)

        # Setting up the range of temperatures for simulated annealing. We start at very
        # high temperatures to approach the random limit. At each temperature, a MC simulation is performed.
        # At the lowest temperatures, you may find that you converge to a ground state.

        temps = np.logspace(4, 2, 10)
        # 函数np.logspace(start, stop, num)返回一个以对数刻度均匀分布的数组，其中 start 是开始的指数，stop 是结束的指数，num 是数组中的元素数量。


        mc_steps = 100000 # Defining number of MC steps at each temperature
        n_thin_by = 10 # Number to thin by

        # Start simulated annealing.
        sampler.anneal(
            temperatures=temps,
            mcmc_steps=mc_steps,
            initial_occupancies=init_occu,
            thin_by=n_thin_by, # Saving every 10 samples
            progress=True # Show the progress bar to know how far along you are
        )

        # Samples are saved in a sample container
        samples = sampler.samples

        print(f'Fraction of successful steps (efficiency) {sampler.efficiency()}')
        print(f'The last step energy is {samples.get_energies()[-1]} eV')
        print(f'The minimum energy in trajectory is {samples.get_minimum_energy()} eV')

        # You can get the minimum energy structure and current structure
        # by using the ensemble processor
        curr_s = ensemble.processor.structure_from_occupancy(samples.get_occupancies()[-1])
        min_s = ensemble.processor.structure_from_occupancy(samples.get_minimum_energy_occupancy())

        n = int(mc_steps / 10)  # number of samples saved for the MC at each temperature
        energies = sampler.samples.get_energies()
        mc_temps = list()  # Create list of temperatures that correspond to the energies

        for t in temps:
            mc_temps.extend([t for i in range(n)])

        # Obtain the average and standard deviation of energy at each temperature.
        for t in temps:
            plot_inds = np.where(mc_temps == t)[0]
            energies_t = np.array([energies[ind] for ind in plot_inds]) / ewald_proc.size
            avg_en = round(np.average(energies_t), 3)
            std_en = round(np.std(energies_t), 4)
            # print(f'At T = {round(t, 2)} K \nAverage energy = {avg_en} eV/prim \nStd dev = {std_en} eV/prim \n')

        lowest_en = sampler.samples.get_minimum_energy() / ewald_proc.size
        lowest_en_occu = sampler.samples.get_minimum_energy_occupancy()
        lowest_en_struct = ensemble.processor.structure_from_occupancy(lowest_en_occu)

        # curr_s.to(filename=out_path,fmt="POSCAR")
        lowest_en_struct.to(filename=out_path, fmt="cif")


    def tm2_json_file(self, input_filename, output_filename):
        with open(input_filename, 'r') as file:
            content = file.read()

        # 将所有 {"element": "Li", "occu": 1} 替换为 {"element": "Li", "oxidation_state": 1.0, "occu": 1}
        content = content.replace('{"element": "Li", "occu": 1}', '{"element": "Li", "oxidation_state": 1.0, "occu": 1}')
        content = content.replace('{"element": "Mn", "occu": 1}', '{"element": "Mn", "oxidation_state": 3.0, "occu": 0.57}, {"element": "Ti", "oxidation_state": 4.0, "occu": 0.43}')
        content = content.replace('{"element": "Ti", "occu": 1}', '{"element": "Mn", "oxidation_state": 3.0, "occu": 0.57}, {"element": "Ti", "oxidation_state": 4.0, "occu": 0.43}')
        content = content.replace('{"element": "O", "occu": 1}', '{"element": "O", "oxidation_state": -2.0, "occu": 1.0}')
        content = content.replace('{"element": "F", "occu": 1}', '{"element": "F", "oxidation_state": -1.0, "occu": 1.0}')

        with open(output_filename, 'w') as file:
            file.write(content)

    def tm2_json_file_onlyEwald(self, input_filename, output_filename):
        with open(input_filename, 'r') as file:
            content = file.read()

        # 将所有 {"element": "Li", "occu": 1} 替换为 {"element": "Li", "oxidation_state": 1.0, "occu": 1}
        content = content.replace('{"element": "Li", "occu": 1}', '{"element": "Li", "oxidation_state": 1.0, "occu": 0.65}, {"element": "Mn", "oxidation_state": 3.0, "occu": 0.2}, {"element": "Ti", "oxidation_state": 4.0, "occu": 0.15}')
        content = content.replace('{"element": "Mn", "occu": 1}', '{"element": "Li", "oxidation_state": 1.0, "occu": 0.65}, {"element": "Mn", "oxidation_state": 3.0, "occu": 0.2}, {"element": "Ti", "oxidation_state": 4.0, "occu": 0.15}')
        content = content.replace('{"element": "Ti", "occu": 1}', '{"element": "Li", "oxidation_state": 1.0, "occu": 0.65}, {"element": "Mn", "oxidation_state": 3.0, "occu": 0.2}, {"element": "Ti", "oxidation_state": 4.0, "occu": 0.15}')
        content = content.replace('{"element": "O", "occu": 1}', '{"element": "O", "oxidation_state": -2.0, "occu": 1.0}')
        content = content.replace('{"element": "F", "occu": 1}', '{"element": "F", "oxidation_state": -1.0, "occu": 1.0}')

        with open(output_filename, 'w') as file:
            file.write(content)


    def tm4_json_file(self, input_filename, output_filename):
        with open(input_filename, 'r') as file:
            content = file.read()

        # 将所有 {"element": "Li", "occu": 1} 替换为 {"element": "Li", "oxidation_state": 1.0, "occu": 1}
        content = content.replace('{"element": "Li", "occu": 1}','{"element": "Li", "oxidation_state": 1.0, "occu": 1}')
        content = content.replace('{"element": "Mn", "occu": 1}','{"element": "Mn", "oxidation_state": 2.0, "occu": 0.2857},{"element": "Mn", "oxidation_state": 3.0, "occu": 0.2857}, {"element": "Ti", "oxidation_state": 4.0, "occu": 0.1429}, {"element": "Nb", "oxidation_state": 5.0, "occu": 0.2857}')
        content = content.replace('{"element": "Ti", "occu": 1}','{"element": "Mn", "oxidation_state": 2.0, "occu": 0.2857},{"element": "Mn", "oxidation_state": 3.0, "occu": 0.2857}, {"element": "Ti", "oxidation_state": 4.0, "occu": 0.1429}, {"element": "Nb", "oxidation_state": 5.0, "occu": 0.2857}')
        content = content.replace('{"element": "Nb", "occu": 1}','{"element": "Mn", "oxidation_state": 2.0, "occu": 0.2857},{"element": "Mn", "oxidation_state": 3.0, "occu": 0.2857}, {"element": "Ti", "oxidation_state": 4.0, "occu": 0.1429}, {"element": "Nb", "oxidation_state": 5.0, "occu": 0.2857}')
        content = content.replace('{"element": "O", "occu": 1}','{"element": "O", "oxidation_state": -2.0, "occu": 1.0}')
        content = content.replace('{"element": "F", "occu": 1}','{"element": "F", "oxidation_state": -1.0, "occu": 1.0}')

        with open(output_filename, 'w') as file:
            file.write(content)


    def tm4_json_file_onlyEwald(self, input_filename, output_filename):
        with open(input_filename, 'r') as file:
            content = file.read()

        # 将所有 {"element": "Li", "occu": 1} 替换为 {"element": "Li", "oxidation_state": 1.0, "occu": 1}
        content = content.replace('{"element": "Li", "occu": 1}','{"element": "Li", "oxidation_state": 1.0, "occu": 0.65}, {"element": "Mn", "oxidation_state": 2.0, "occu": 0.1},{"element": "Mn", "oxidation_state": 3.0, "occu": 0.1}, {"element": "Ti", "oxidation_state": 4.0, "occu": 0.05}, {"element": "Nb", "oxidation_state": 5.0, "occu": 0.1}')
        content = content.replace('{"element": "Mn", "occu": 1}','{"element": "Li", "oxidation_state": 1.0, "occu": 0.65}, {"element": "Mn", "oxidation_state": 2.0, "occu": 0.1},{"element": "Mn", "oxidation_state": 3.0, "occu": 0.1}, {"element": "Ti", "oxidation_state": 4.0, "occu": 0.05}, {"element": "Nb", "oxidation_state": 5.0, "occu": 0.1}')
        content = content.replace('{"element": "Ti", "occu": 1}','{"element": "Li", "oxidation_state": 1.0, "occu": 0.65}, {"element": "Mn", "oxidation_state": 2.0, "occu": 0.1},{"element": "Mn", "oxidation_state": 3.0, "occu": 0.1}, {"element": "Ti", "oxidation_state": 4.0, "occu": 0.05}, {"element": "Nb", "oxidation_state": 5.0, "occu": 0.1}')
        content = content.replace('{"element": "Nb", "occu": 1}','{"element": "Li", "oxidation_state": 1.0, "occu": 0.65}, {"element": "Mn", "oxidation_state": 2.0, "occu": 0.1},{"element": "Mn", "oxidation_state": 3.0, "occu": 0.1}, {"element": "Ti", "oxidation_state": 4.0, "occu": 0.05}, {"element": "Nb", "oxidation_state": 5.0, "occu": 0.1}')
        content = content.replace('{"element": "O", "occu": 1}','{"element": "O", "oxidation_state": -2.0, "occu": 1.0}')
        content = content.replace('{"element": "F", "occu": 1}','{"element": "F", "oxidation_state": -1.0, "occu": 1.0}')

        with open(output_filename, 'w') as file:
            file.write(content)

    def tm6_json_file(self, input_filename, output_filename):
        with open(input_filename, 'r') as file:
            content = file.read()

        # 将所有 {"element": "Li", "occu": 1} 替换为 {"element": "Li", "oxidation_state": 1.0, "occu": 1}
        content = content.replace('{"element": "Li", "occu": 1}','{"element": "Li", "oxidation_state": 1.0, "occu": 1}')
        content = content.replace('{"element": "Mn", "occu": 1}','{"element": "Mn", "oxidation_state": 2.0, "occu": 0.143},{"element": "Mn", "oxidation_state": 3.0, "occu": 0.143}, {"element": "Co", "oxidation_state": 2.0, "occu": 0.143},{"element": "Cr", "oxidation_state": 3.0, "occu": 0.143},{"element": "Ti", "oxidation_state": 4.0, "occu": 0.143}, {"element": "Nb", "oxidation_state": 5.0, "occu": 0.285}')
        content = content.replace('{"element": "Co", "occu": 1}','{"element": "Mn", "oxidation_state": 2.0, "occu": 0.143},{"element": "Mn", "oxidation_state": 3.0, "occu": 0.143}, {"element": "Co", "oxidation_state": 2.0, "occu": 0.143},{"element": "Cr", "oxidation_state": 3.0, "occu": 0.143},{"element": "Ti", "oxidation_state": 4.0, "occu": 0.143}, {"element": "Nb", "oxidation_state": 5.0, "occu": 0.285}')
        content = content.replace('{"element": "Cr", "occu": 1}','{"element": "Mn", "oxidation_state": 2.0, "occu": 0.143},{"element": "Mn", "oxidation_state": 3.0, "occu": 0.143}, {"element": "Co", "oxidation_state": 2.0, "occu": 0.143},{"element": "Cr", "oxidation_state": 3.0, "occu": 0.143},{"element": "Ti", "oxidation_state": 4.0, "occu": 0.143}, {"element": "Nb", "oxidation_state": 5.0, "occu": 0.285}')
        content = content.replace('{"element": "Ti", "occu": 1}','{"element": "Mn", "oxidation_state": 2.0, "occu": 0.143},{"element": "Mn", "oxidation_state": 3.0, "occu": 0.143}, {"element": "Co", "oxidation_state": 2.0, "occu": 0.143},{"element": "Cr", "oxidation_state": 3.0, "occu": 0.143},{"element": "Ti", "oxidation_state": 4.0, "occu": 0.143}, {"element": "Nb", "oxidation_state": 5.0, "occu": 0.285}')
        content = content.replace('{"element": "Nb", "occu": 1}','{"element": "Mn", "oxidation_state": 2.0, "occu": 0.143},{"element": "Mn", "oxidation_state": 3.0, "occu": 0.143}, {"element": "Co", "oxidation_state": 2.0, "occu": 0.143},{"element": "Cr", "oxidation_state": 3.0, "occu": 0.143},{"element": "Ti", "oxidation_state": 4.0, "occu": 0.143}, {"element": "Nb", "oxidation_state": 5.0, "occu": 0.285}')
        content = content.replace('{"element": "O", "occu": 1}','{"element": "O", "oxidation_state": -2.0, "occu": 1.0}')
        content = content.replace('{"element": "F", "occu": 1}','{"element": "F", "oxidation_state": -1.0, "occu": 1.0}')

        with open(output_filename, 'w') as file:
            file.write(content)

    def tm6_json_file_onlyEwald(self, input_filename, output_filename):
        with open(input_filename, 'r') as file:
            content = file.read()

        # 将所有 {"element": "Li", "occu": 1} 替换为 {"element": "Li", "oxidation_state": 1.0, "occu": 1}
        content = content.replace('{"element": "Li", "occu": 1}','{"element": "Li", "oxidation_state": 1.0, "occu": 0.65}, {"element": "Mn", "oxidation_state": 2.0, "occu": 0.05},{"element": "Mn", "oxidation_state": 3.0, "occu": 0.05}, {"element": "Co", "oxidation_state": 2.0, "occu": 0.05},{"element": "Cr", "oxidation_state": 3.0, "occu": 0.05},{"element": "Ti", "oxidation_state": 4.0, "occu": 0.05}, {"element": "Nb", "oxidation_state": 5.0, "occu": 0.1}')
        content = content.replace('{"element": "Mn", "occu": 1}','{"element": "Li", "oxidation_state": 1.0, "occu": 0.65}, {"element": "Mn", "oxidation_state": 2.0, "occu": 0.05},{"element": "Mn", "oxidation_state": 3.0, "occu": 0.05}, {"element": "Co", "oxidation_state": 2.0, "occu": 0.05},{"element": "Cr", "oxidation_state": 3.0, "occu": 0.05},{"element": "Ti", "oxidation_state": 4.0, "occu": 0.05}, {"element": "Nb", "oxidation_state": 5.0, "occu": 0.1}')
        content = content.replace('{"element": "Co", "occu": 1}','{"element": "Li", "oxidation_state": 1.0, "occu": 0.65}, {"element": "Mn", "oxidation_state": 2.0, "occu": 0.05},{"element": "Mn", "oxidation_state": 3.0, "occu": 0.05}, {"element": "Co", "oxidation_state": 2.0, "occu": 0.05},{"element": "Cr", "oxidation_state": 3.0, "occu": 0.05},{"element": "Ti", "oxidation_state": 4.0, "occu": 0.05}, {"element": "Nb", "oxidation_state": 5.0, "occu": 0.1}')
        content = content.replace('{"element": "Cr", "occu": 1}','{"element": "Li", "oxidation_state": 1.0, "occu": 0.65}, {"element": "Mn", "oxidation_state": 2.0, "occu": 0.05},{"element": "Mn", "oxidation_state": 3.0, "occu": 0.05}, {"element": "Co", "oxidation_state": 2.0, "occu": 0.05},{"element": "Cr", "oxidation_state": 3.0, "occu": 0.05},{"element": "Ti", "oxidation_state": 4.0, "occu": 0.05}, {"element": "Nb", "oxidation_state": 5.0, "occu": 0.1}')
        content = content.replace('{"element": "Ti", "occu": 1}','{"element": "Li", "oxidation_state": 1.0, "occu": 0.65}, {"element": "Mn", "oxidation_state": 2.0, "occu": 0.05},{"element": "Mn", "oxidation_state": 3.0, "occu": 0.05}, {"element": "Co", "oxidation_state": 2.0, "occu": 0.05},{"element": "Cr", "oxidation_state": 3.0, "occu": 0.05},{"element": "Ti", "oxidation_state": 4.0, "occu": 0.05}, {"element": "Nb", "oxidation_state": 5.0, "occu": 0.1}')
        content = content.replace('{"element": "Nb", "occu": 1}','{"element": "Li", "oxidation_state": 1.0, "occu": 0.65}, {"element": "Mn", "oxidation_state": 2.0, "occu": 0.05},{"element": "Mn", "oxidation_state": 3.0, "occu": 0.05}, {"element": "Co", "oxidation_state": 2.0, "occu": 0.05},{"element": "Cr", "oxidation_state": 3.0, "occu": 0.05},{"element": "Ti", "oxidation_state": 4.0, "occu": 0.05}, {"element": "Nb", "oxidation_state": 5.0, "occu": 0.1}')
        content = content.replace('{"element": "O", "occu": 1}','{"element": "O", "oxidation_state": -2.0, "occu": 1.0}')
        content = content.replace('{"element": "F", "occu": 1}','{"element": "F", "oxidation_state": -1.0, "occu": 1.0}')

        with open(output_filename, 'w') as file:
            file.write(content)


