from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.local_env import BrunnerNN_real
from pymatgen.core.structure import Structure, Element
import numpy as np
from typing import Union
import random
import math


class SRO:
    def __init__(
            self,
            structure_path: str,
            anion_a: str = "F",
            anion_b: str = "O",
            cation: str = "Li",
            c_cation: Union[int, float] = 26 / 40,  # Concentration of cation(Li)
    ):
        self.structure = Structure.from_file(structure_path)
        self.anion_a = anion_a
        self.anion_b = anion_b
        self.cation = cation
        self.c_cation = c_cation
        self.a_idxs = self.get_idxs("F")  # Obtain all index numbers of F
        self.b_idxs = self.get_idxs("O")
        self.all_cation_idxs = list(set(range(self.structure.num_sites)) - set(self.a_idxs) - set(self.b_idxs))
        self.cnn = CrystalNN()
        self.bnn = BrunnerNN_real()
        self.anion_cn = 6  # Anion coordination number, which is 6 in DRX
        self.cation_cn = 12  # cation-cation coordination number, which is 12 in DRX.
        self.a, self.a_dict = self.alpha()
        self.a_LiLi, self.a_LiLi_dict = self.alpha_LiLi()

    def get_idxs(self, a: str):
        """
        Obtain index numbers of element a.
        """
        i_idxs = []
        for i in range(self.structure.num_sites):
            if self.structure.species[i] == Element(a):
                i_idxs.append(i)
        return i_idxs

    def alpha_fix(self):
        """
        Calculate alpha where anion is the i species, cation is the j species, as defined in PNAS, 2021, 118, e2020540118.
        """
        anion_idxs = self.get_idxs(self.anion_a)
        alpha_list = []
        for i in anion_idxs:
            # P is the probability of finding cation Li adjacent to anion F
            # alpha shuold be the average value of all the anions F
            P = self.cnn.get_cn_dict(self.structure, i).get(self.cation, 0) / self.anion_cn
            alpha_list.append(1 - P / self.c_cation)
        return np.mean(alpha_list)

    def alpha(self):
        """
        Calculate alpha where anion is the i species, cation is the j species, as defined in PNAS, 2021, 118, e2020540118.
        """
        # anion_idxs = self.get_idxs(self.anion_a)
        anion_idxs = self.a_idxs
        alpha_dic = dict()
        for i in anion_idxs:
            # P is the probability of finding cation Li adjacent to anion F
            # alpha shuold be the average value of all the anions F
            P = self.cnn.get_cn_dict(self.structure, i).get(self.cation, 0) / self.anion_cn
            alpha_dic[i] = (1 - P / self.c_cation)
        alpha = sum(alpha_dic.values()) / len(alpha_dic)

        return alpha, alpha_dic

    def alpha_new(self, a_neighbor: int, b_neighbor: int):
        changed_anion_ls = []
        for i in range(6):
            if self.cnn.get_nn(self.structure, a_neighbor)[i].specie == Element('F'):
                changed_anion_ls.append(self.cnn.get_nn(self.structure, a_neighbor)[i].index)
            if self.cnn.get_nn(self.structure, b_neighbor)[i].specie == Element('F'):
                changed_anion_ls.append(self.cnn.get_nn(self.structure, b_neighbor)[i].index)

        new_dict = self.a_dict.copy()

        for i in changed_anion_ls:
            P = self.cnn.get_cn_dict(self.structure, i).get(self.cation, 0) / self.anion_cn
            new_dict[i] = (1 - P / self.c_cation)

        new_alpha = sum(new_dict.values()) / len(new_dict)
        # print("Changed new_alpha:", new_alpha)
        return new_alpha, new_dict

    def alpha_LiLi_fix(self):
        """
        Calculate alpha_LiLi.
        """
        structrue_dup = self.structure.copy()
        last_idx = int(self.structure.num_sites - 1)
        mid_idx = int(self.structure.num_sites / 2 - 1)
        for i in range(last_idx, mid_idx, -1):
            structrue_dup.pop(i)

        cation_idxs = self.get_idxs(self.cation)
        alpha_LiLi_list = []

        for i in cation_idxs:
            # P is the probability of finding second nearest neighbor cation Li adjacent to cation Li.
            # The second nearest neighbor is a cation - the coordination number of the cation is 12
            P = self.bnn.get_cn_dict(structrue_dup, i).get(self.cation, 0) / self.cation_cn
            alpha_LiLi_list.append(1 - P / self.c_cation)

        return np.mean(alpha_LiLi_list)

    def alpha_LiLi(self):
        """
        Calculate alpha_LiLi and alpha_LiLi_dict.
        """
        structrue_dup = self.structure.copy()
        last_idx = int(self.structure.num_sites - 1)
        mid_idx = int(self.structure.num_sites / 2 - 1)
        for i in range(last_idx, mid_idx, -1):
            structrue_dup.pop(i)

        cation_idxs = self.get_idxs(self.cation)
        alphalili_dic = dict()

        for i in cation_idxs:
            # P is the probability of finding second nearest neighbor cation Li adjacent to cation Li.
            # The second nearest neighbor is a cation - the coordination number of the cation is 12
            P = self.bnn.get_cn_dict(structrue_dup, i).get(self.cation, 0) / self.cation_cn
            alphalili_dic[i] = (1 - P / self.c_cation)

        alpha_LiLi = sum(alphalili_dic.values()) / len(alphalili_dic)

        return alpha_LiLi, alphalili_dic

    def alpha_LiLi_new(self, c_neighbor: int, d_neighbor: int, c_is_Li: bool):
        structrue_dup = self.structure.copy()
        last_idx = int(self.structure.num_sites - 1)
        mid_idx = int(self.structure.num_sites / 2 - 1)
        for i in range(last_idx, mid_idx, -1):
            structrue_dup.pop(i)

        changed_cation_ls = []
        for i in range(12):
            if self.bnn.get_nn(structrue_dup, c_neighbor)[i].specie == Element('Li'):
                changed_cation_ls.append(self.bnn.get_nn(structrue_dup, c_neighbor)[i].index)
            if self.bnn.get_nn(structrue_dup, d_neighbor)[i].specie == Element('Li'):
                changed_cation_ls.append(self.bnn.get_nn(structrue_dup, d_neighbor)[i].index)

        new_dict_LiLi = self.a_LiLi_dict.copy()

        if c_is_Li:
            new_dict_LiLi[d_neighbor] = 0
            new_dict_LiLi.pop(c_neighbor)
            changed_cation_ls.append(d_neighbor)
        else:
            new_dict_LiLi[c_neighbor] = 0
            new_dict_LiLi.pop(d_neighbor)
            changed_cation_ls.append(c_neighbor)

        for i in changed_cation_ls:
            P = self.bnn.get_cn_dict(structrue_dup, i).get(self.cation, 0) / self.cation_cn
            new_dict_LiLi[i] = (1 - P / self.c_cation)

        new_alpha_LiLi = sum(new_dict_LiLi.values()) / len(new_dict_LiLi)
        # print("Changed new_alpha:", new_alpha_LiLi)
        return new_alpha_LiLi, new_dict_LiLi

    def get_neighbor(self, center: str, around: str):
        """
        Calculate the average number of nearest-neighbor element-a around element-b.
        """
        li_f_cn = []
        for i in range(self.structure.num_sites):
            if self.structure.species[i] == Element(center):
                li_f_cn.append(self.bnn.get_cn_dict(self.structure, i).get(around, 0))
        return np.mean(li_f_cn)

    def get_neighbor_LiLi(self):
        """
        Calculate the average number of second nearest-neighbor Li around Li.
        """
        OF_list = []
        last_idx = int(self.structure.num_sites - 1)
        mid_idx = int(self.structure.num_sites / 2 - 1)
        for i in range(last_idx, mid_idx, -1):
            OF_list.append(self.structure[i])
            self.structure.pop(i)

        li_li = self.get_neighbor(self.cation, self.cation)
        for i in range(len(OF_list)):
            self.structure.append(OF_list[i].specie, OF_list[i].frac_coords)
        return li_li

    def get_Li_2NN_environment(self, idx: int):
        """
        Calculate the second nearest-neighbor cations distribution around Li.
        """
        structrue_dup = self.structure.copy()
        last_idx = int(self.structure.num_sites - 1)
        mid_idx = int(self.structure.num_sites / 2 - 1)
        for i in range(last_idx, mid_idx, -1):
            structrue_dup.pop(i)
        env = self.bnn.get_nn(structrue_dup, idx)

        return env

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid function.
        """
        return 1 / (1 + math.exp(-x))

    def exchange_site(self, a: int, b: int):
        """
        Exchange element on two sites.
        """
        try:
            original_a = self.structure.species[a].to_pretty_string()
        except AttributeError:
            original_a = self.structure.species[a].name
        try:
            original_b = self.structure.species[b].to_pretty_string()
        except AttributeError:
            original_b = self.structure.species[b].name
        self.structure[b] = original_a
        self.structure[a] = original_b

    def exchange(
            self,
            target_alpha: Union[int, float] = 0,
            rate: Union[int, float] = 1,
            random_seed: Union[None, int] = None
    ):
        """
        Perform exchange of Li and M once.
        """
        random.seed(random_seed)
        diff = self.a - target_alpha
        prob = self.sigmoid(diff * rate)

        a_site = self.a_idxs[random.randrange(len(self.a_idxs))]
        b_site = self.b_idxs[random.randrange(len(self.b_idxs))]
        target = True
        m = 0
        while True:
            a_neighbor = int(self.cnn.get_nn(self.structure, a_site)[random.randrange(self.anion_cn)].index)
            b_neighbor = int(self.cnn.get_nn(self.structure, b_site)[random.randrange(self.anion_cn)].index)
            if (self.structure.species[a_neighbor] == Element(self.cation) and self.structure.species[
                b_neighbor] != Element(self.cation)):
                break
            elif self.structure.species[a_neighbor] != Element(self.cation) and self.structure.species[
                b_neighbor] == Element(self.cation):
                target = False
                break
            m += 1
            if m == 100:
                self.a = self.alpha_fix()
                print("Cannot find correct cations")
                return self.a

        # print(a_neighbor, b_neighbor)
        old_alpha = self.a
        if random.random() < prob:
            if not target:  
                self.exchange_site(a_neighbor, b_neighbor)
                new_alpha, new_dict = self.alpha_new(a_neighbor, b_neighbor)
                if abs(new_alpha - target_alpha) <= abs(old_alpha - target_alpha):
                    print("More neighboring")
                    self.a = new_alpha
                    self.a_dict = new_dict
                else:
                    self.exchange_site(a_neighbor, b_neighbor)
                    print("No exchange")
                    self.a = old_alpha
            else:
                print("No exchange")
        else:
            if target:  
                self.exchange_site(a_neighbor, b_neighbor)
                new_alpha, new_alpdict = self.alpha_new(a_neighbor, b_neighbor)
                if abs(new_alpha - target_alpha) <= abs(old_alpha - target_alpha):
                    print("Less neighboring")  # new_alpha >= old_alpha
                    self.a = new_alpha
                    self.a_dict = new_alpdict
                else:
                    self.exchange_site(a_neighbor, b_neighbor)
                    print("No exchange")
                    self.a = old_alpha
            else:
                print("No exchange")
        return self.a

    def exchange_LiLi(
            self,
            target_alpha_LiLi: Union[int, float] = 0,
            target_alpha: Union[int, float] = 0,
            tol: Union[int, float] = 0.05,
            rate: Union[int, float] = 1,
            random_seed: Union[None, int] = None
    ):
        """
        Perform exchange of Li and M around Li once.
        """
        random.seed(random_seed)
        diff = self.a_LiLi - target_alpha_LiLi
        prob = self.sigmoid(diff * rate)

        c_idxs = self.get_idxs("Li")
        d_idxs = list(set(self.all_cation_idxs) - set(c_idxs))
        c_site = c_idxs[random.randrange(len(c_idxs))]  # "c_site" represents the position of any arbitrary Li atom.
        d_site = d_idxs[random.randrange(len(d_idxs))]  # "d_site" represents the position of any arbitrary TM atom.

        target = True
        m = 0
        while True:
            c_neighbor = int(self.get_Li_2NN_environment(c_site)[random.randrange(self.cation_cn)].index)
            d_neighbor = int(self.get_Li_2NN_environment(d_site)[random.randrange(self.cation_cn)].index)
            if (self.structure.species[c_neighbor] == Element(self.cation) and self.structure.species[
                d_neighbor] != Element(self.cation)):
                break
                # That is, if the selection around Li is Li and the selection around TM is TM, then keep target=True and proceed to the next step.
            elif self.structure.species[c_neighbor] != Element(self.cation) and self.structure.species[
                d_neighbor] == Element(self.cation):
                target = False
                # That is, if the selection around Li is TM and the selection around TM is Li, then set target to False and proceed to the next step.
                break

            m += 1
            if m == 100:
                self.a_LiLi = self.alpha_LiLi_fix()
                print("Cannot find correct cations")
                return self.a_LiLi

        old_alpha_LiLi = self.a_LiLi
        if random.random() < prob:  # The amount of Li surrounding the structure Li is less than the target value.
            if not target:  # This corresponds to target=False, meaning that in this case, Li was not selected around the Li area, but Li was selected around the TM area.
                self.exchange_site(c_neighbor, d_neighbor)
                new_alpha_LiLi, new_dict_LiLi = self.alpha_LiLi_new(c_neighbor, d_neighbor, False)
                new_alpha, new_dict = self.alpha_new(c_neighbor, d_neighbor)
                if abs(new_alpha_LiLi - target_alpha_LiLi) <= abs(old_alpha_LiLi - target_alpha_LiLi) and abs(
                        new_alpha - target_alpha) <= tol:
                    self.a_LiLi = new_alpha_LiLi
                    self.a_LiLi_dict = new_dict_LiLi
                    self.a = new_alpha
                    self.a_dict = new_dict
                    print("More neighboring LiLi")
                else:
                    self.exchange_site(c_neighbor, d_neighbor)
                    print("No exchange")
            else:
                print("No exchange")
        else:  # The amount of Li surrounding the structure Li is higher than the target value.
            if target:  # This corresponds to target=True, meaning that in the Li area, Li is selected, while in the TM area, Li is not selected.
                self.exchange_site(c_neighbor, d_neighbor)
                new_alpha_LiLi, new_dict_LiLi = self.alpha_LiLi_new(c_neighbor, d_neighbor, True)
                new_alpha, new_dict = self.alpha_new(c_neighbor, d_neighbor)
                if abs(new_alpha_LiLi - target_alpha_LiLi) <= abs(old_alpha_LiLi - target_alpha_LiLi) and abs(
                        new_alpha - target_alpha) <= tol:
                    self.a_LiLi = new_alpha_LiLi
                    self.a_LiLi_dict = new_dict_LiLi
                    self.a = new_alpha
                    self.a_dict = new_dict
                    print("less neighboring LiLi")
                else:
                    self.exchange_site(c_neighbor, d_neighbor)
                    print("No exchange")
            else:
                print("No exchange")
        return self.a_LiLi

    def run_lif(self, max_steps: int,
            target_alpha: Union[int, float] = 0,
            # target_alpha_LiLi: Union[int, float] = 0,
            rate: Union[int, float] = 1,
            tol: Union[int, float] = 0.05,
            random_seed: Union[None, int] = None,
            ):

        random.seed(random_seed)
        print("Innitial alpha:", self.a, "Innitial alphaLiLi:", self.a_LiLi)
        for i in range(max_steps):
            if abs(self.a - target_alpha) <= tol:
                # print("Target alpha reached")
                break
            self.exchange(target_alpha, rate, random_seed=random.randrange(1000))
            print("Steps：", i, "New alpha:", self.a)

        if abs(self.alpha_fix() - target_alpha) <= tol:
            print("target alpha_LiF reached.")
        else:
            print("Target alpha_LiF not reached")

    def run(self, max_steps: int,
            target_alpha: Union[int, float] = 0,
            target_alpha_LiLi: Union[int, float] = 0,
            rate: Union[int, float] = 1,
            tol: Union[int, float] = 0.05,
            random_seed: Union[None, int] = None,
            ):

        random.seed(random_seed)
        print("Innitial alpha:", self.a, "Innitial alphaLiLi:", self.a_LiLi)
        for i in range(max_steps):
            if abs(self.a - target_alpha) <= tol:
                print("Target alpha_LiF reached")
                break
            self.exchange(target_alpha, rate, random_seed=random.randrange(1000))
            print("Steps：", i, "New alpha:", self.a)

        self.a_LiLi, self.a_LiLi_dict = self.alpha_LiLi()
        if abs(self.alpha_fix() - target_alpha) <= tol:
            for i in range(max_steps):
                if abs(self.a_LiLi - target_alpha_LiLi) <= tol:
                    print("Target alpha_LiLi reached")
                    break
                self.exchange_LiLi(target_alpha_LiLi, target_alpha, tol, rate, random_seed=random.randrange(1000))
                print("Steps：", i, "New alpha_LiLi:", self.a_LiLi)
        else:
            print("Target alpha_LiF not reached")

        if abs(self.alpha_LiLi_fix() - target_alpha_LiLi) <= tol:
            print("Both target alpha and alpha_LiLi reached.")
        else:
            print("Target alpha_LiLi not reached")
            

    def to_file(self, path):
        self.structure.to(filename=path, fmt='poscar')


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
            tm_type: str,
    ):
        input_str = Structure.from_file(structure_path)
        dumpfn(input_str, 'input_file.json')

        if tm_type == "TM2":
            self.tm2_json_file('input_file.json', 'input_file1.json')
        elif tm_type == "TM4":
            self.tm4_json_file('input_file.json', 'input_file1.json')
        elif tm_type == "TM6":
            self.tm6_json_file('input_file.json', 'input_file1.json')
        else:
            raise ValueError(f"Unsupported tm_type: {tm_type}. Use 'TM2', 'TM4', or 'TM6'.")
            
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
        print(f'The supercell size for the processor is {ensemble.processor.size} prims.')
        print(f'The ensemble has a total of {ensemble.num_sites} sites.')
        print(f'The active sublattices are:')
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
        #temps = np.array([2000])

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

        # print(f'Fraction of successful steps (efficiency) {sampler.efficiency()}')
        # print(f'The last step energy is {samples.get_energies()[-1]} eV')
        # print(f'The minimum energy in trajectory is {samples.get_minimum_energy()} eV')

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

        lowest_en_struct.to(filename=out_path,fmt="POSCAR")

    
    def tm2_json_file(self, input_filename, output_filename):
        with open(input_filename, 'r') as file:
            content = file.read()

        # Replace all {"element": "Li", "occu": 1} with {"element": "Li", "oxidation_state": 1.0, "occu": 1}
        content = content.replace('{"element": "Li", "occu": 1}', '{"element": "Li", "oxidation_state": 1.0, "occu": 1}')
        content = content.replace('{"element": "Mn", "occu": 1}', '{"element": "Mn", "oxidation_state": 3.0, "occu": 0.571}, {"element": "Ti", "oxidation_state": 4.0, "occu": 0.429}')
        content = content.replace('{"element": "Ti", "occu": 1}', '{"element": "Mn", "oxidation_state": 3.0, "occu": 0.571}, {"element": "Ti", "oxidation_state": 4.0, "occu": 0.429}')
        content = content.replace('{"element": "O", "occu": 1}', '{"element": "O", "oxidation_state": -2.0, "occu": 1.0}')
        content = content.replace('{"element": "F", "occu": 1}', '{"element": "F", "oxidation_state": -1.0, "occu": 1.0}')

        with open(output_filename, 'w') as file:
            file.write(content)

    def tm4_json_file(self, input_filename, output_filename):
        with open(input_filename, 'r') as file:
            content = file.read()

        # Replace all {"element": "Li", "occu": 1} with {"element": "Li", "oxidation_state": 1.0, "occu": 1}
        content = content.replace('{"element": "Li", "occu": 1}','{"element": "Li", "oxidation_state": 1.0, "occu": 1}')
        content = content.replace('{"element": "Mn", "occu": 1}','{"element": "Mn", "oxidation_state": 2.0, "occu": 0.2857},{"element": "Mn", "oxidation_state": 3.0, "occu": 0.2857}, {"element": "Ti", "oxidation_state": 4.0, "occu": 0.1429}, {"element": "Nb", "oxidation_state": 5.0, "occu": 0.2857}')
        content = content.replace('{"element": "Ti", "occu": 1}','{"element": "Mn", "oxidation_state": 2.0, "occu": 0.2857},{"element": "Mn", "oxidation_state": 3.0, "occu": 0.2857}, {"element": "Ti", "oxidation_state": 4.0, "occu": 0.1429}, {"element": "Nb", "oxidation_state": 5.0, "occu": 0.2857}')
        content = content.replace('{"element": "Nb", "occu": 1}','{"element": "Mn", "oxidation_state": 2.0, "occu": 0.2857},{"element": "Mn", "oxidation_state": 3.0, "occu": 0.2857}, {"element": "Ti", "oxidation_state": 4.0, "occu": 0.1429}, {"element": "Nb", "oxidation_state": 5.0, "occu": 0.2857}')
        content = content.replace('{"element": "O", "occu": 1}','{"element": "O", "oxidation_state": -2.0, "occu": 1.0}')
        content = content.replace('{"element": "F", "occu": 1}','{"element": "F", "oxidation_state": -1.0, "occu": 1.0}')

        with open(output_filename, 'w') as file:
            file.write(content)

    def tm6_json_file(self, input_filename, output_filename):
        with open(input_filename, 'r') as file:
            content = file.read()

        # Replace all {"element": "Li", "occu": 1} with {"element": "Li", "oxidation_state": 1.0, "occu": 1}
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