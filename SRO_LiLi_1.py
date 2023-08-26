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
        self.c_idxs = self.get_idxs("Li")  # self.d_idxs are the indexs of TM
        self.d_idxs = list(set(list(range(0, self.structure.num_sites))) - set(self.a_idxs) - set(self.b_idxs) - set(self.c_idxs))
        self.cnn = CrystalNN()
        self.bnn = BrunnerNN_real()
        self.anion_cn = self.cnn.get_cn(self.structure, -1)  # Anion coordination number, which is 6 in DRX
        self.cation_cn = 12  # cation-cation coordination number, which is 12 in DRX.
        self.a = self.alpha()
        self.a_LiLi = self.alpha_LiLi()  # alpha_LiLi

    def get_idxs(self, a: str):
        """
        Obtain index numbers of element a.
        """
        i_idxs = []
        for i in range(self.structure.num_sites):
            if self.structure.species[i] == Element(a):
                i_idxs.append(i)
        return i_idxs

    def alpha(self):
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
        prob = self.sigmoid(diff * rate)  # probability
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
                self.a = self.alpha()
                return self.a
        if random.random() < prob:  # a site coordinate with target cation
            if not target:
                self.exchange_site(a_neighbor, b_neighbor)
                print("More neighboring")  # new_alpha <= old_alpha
                self.a = self.alpha()
        else:  # a site not coordinate with target cation
            if target:
                self.exchange_site(a_neighbor, b_neighbor)
                print("Less neighboring")  # new_alpha >= old_alpha
                self.a = self.alpha()
        return self.a

    def run(self, max_steps: int,
            target_alpha: Union[int, float] = 0,
            rate: Union[int, float] = 1,
            tol: Union[int, float] = 0.05,
            random_seed: Union[None, int] = None,
            ):
        old_alpha = self.a
        for i in range(max_steps):
            self.exchange(target_alpha, rate, random_seed)
            print(self.a)
            if old_alpha == self.a and abs(self.a - target_alpha) <= tol:
                return "Target alpha reached"
            old_alpha = self.a
        return "Target alpha not reached"

    def to_file(self, path):
        self.structure.to(filename=path)