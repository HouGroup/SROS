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
        self.a_idxs = self.get_idxs("F")  # Obtain all the index numbers of F
        self.b_idxs = self.get_idxs("O")
        self.c_idxs = self.get_idxs("Li")  # self.d_idxs are the indexs of TM
        self.d_idxs = list(set(list(range(0, self.structure.num_sites))) - set(self.a_idxs) - set(self.b_idxs) - set(self.c_idxs))
        self.cnn = CrystalNN()
        self.bnn = BrunnerNN_real()
        self.anion_cn = self.cnn.get_cn(self.structure,-1)  # Anion coordination number, which is 6 in DRX. -1 refers to  the last atom.
        self.cation_cn = 12  # cation-cation coordination number, which is 12 in DRX.
        self.a = self.alpha()  # alpha_FLi
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
            P = self.cnn.get_cn_dict(self.structure, i).get(self.cation, 0) / self.anion_cn
            alpha_list.append(1 - P / self.c_cation)
        return np.mean(alpha_list)

    def alpha_LiLi(self):
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
            # 第二近邻为阳离子-阳离子配位数为12
            P = self.bnn.get_cn_dict(structrue_dup, i).get(self.cation, 0) / self.cation_cn
            alpha_LiLi_list.append(1 - P / self.c_cation)

        return np.mean(alpha_LiLi_list)

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
                self.a = self.alpha()
                return self.a

        if random.random() < prob:
            if not target:
                self.exchange_site(a_neighbor, b_neighbor)
                print("More neighboring F-Li")
                self.a = self.alpha()
            else:
                print("No exchange")
        else:
            if target:
                self.exchange_site(a_neighbor, b_neighbor)
                print("Less neighboring F-Li")
                self.a = self.alpha()
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

        c_site = self.c_idxs[random.randrange(len(self.c_idxs))]  # c_site是任意一个Li的位置
        d_site = self.d_idxs[random.randrange(len(self.d_idxs))]  # d_site是任意一个TM的位置

        target = True
        m = 0
        while True:
            c_neighbor = int(self.get_Li_2NN_environment(c_site)[random.randrange(self.cation_cn)].index)
            d_neighbor = int(self.get_Li_2NN_environment(d_site)[random.randrange(self.cation_cn)].index)
            if (self.structure.species[c_neighbor] == Element(self.cation) and self.structure.species[
                d_neighbor] != Element(self.cation)):
                break
                # 即如果Li周围选中的是Li，TM周围选中的是TM，则保持target=True并进入下一步
            elif self.structure.species[c_neighbor] != Element(self.cation) and self.structure.species[
                d_neighbor] == Element(self.cation):
                target = False
                # 即如果Li周围选中的是TM，TM周围选中的是Li，则使target=False并进入下一步
                break

            m += 1
            if m == 100:
                self.a_LiLi = self.alpha_LiLi()
                return self.a_LiLi

        old_alpha_LiLi = self.alpha_LiLi()
        if random.random() < prob:  # 结构Li周围的Li比目标值少
            if not target:  # 这里对应的是target=False，即Li周围选的不是Li，而TM周围选的是Li
                self.exchange_site(c_neighbor, d_neighbor)
                new_alpha_LiLi = self.alpha_LiLi()
                if abs(new_alpha_LiLi - target_alpha_LiLi) < abs(old_alpha_LiLi - target_alpha_LiLi) and abs(
                        self.alpha() - target_alpha) <= tol:
                    print("More neighboring LiLi")
                else:
                    self.exchange_site(c_neighbor, d_neighbor)
                    print("No exchange")
            else:
                print("No exchange")
        else:  # 结构Li周围的Li比目标值多
            if target:  # 这里对应的是target=True，即Li周围选的是Li，而TM周围选的不是Li
                self.exchange_site(c_neighbor, d_neighbor)
                new_alpha_LiLi = self.alpha_LiLi()
                if abs(new_alpha_LiLi - target_alpha_LiLi) < abs(old_alpha_LiLi - target_alpha_LiLi) and abs(
                        self.alpha() - target_alpha) <= tol:
                    print("less neighboring LiLi")
                else:
                    self.exchange_site(c_neighbor, d_neighbor)
                    print("No exchange")
            else:
                print("No exchange")
        return self.a_LiLi

    def run(self, max_steps: int,
            target_alpha: Union[int, float] = 0,
            target_alpha_LiLi: Union[int, float] = 0,
            rate: Union[int, float] = 1,
            tol: Union[int, float] = 0.05,
            random_seed: Union[None, int] = None,
            ):
        random.seed(random_seed)
        print("Innitial alpha_FLi:", self.alpha(), "Innitial F-Li:", self.get_neighbor(self.anion_a, self.cation))
        print("Innitial alpha_LiLi:", self.alpha_LiLi(), "Innitial Li-Li:", self.get_neighbor_LiLi())
        for i in range(max_steps):
            if abs(self.alpha() - target_alpha) <= tol:
                print("Final F-Li:", self.get_neighbor(self.anion_a, self.cation), "Final Li-Li:",
                      self.get_neighbor_LiLi())
                print("Target alpha reached")
                break
            self.exchange(target_alpha, rate, random_seed=random.randrange(1000))
            print("New alpha_FLi:", self.alpha(), "F-Li:", self.get_neighbor(self.anion_a, self.cation))
            print("New alpha_LiLi:", self.alpha_LiLi(), "Li-Li:", self.get_neighbor_LiLi())

        if abs(self.alpha() - target_alpha) <= tol:
            for i in range(max_steps):
                self.exchange_LiLi(target_alpha_LiLi, target_alpha, tol, rate, random_seed=random.randrange(1000))
                print("New alpha_FLi:", self.alpha(), "F-Li:", self.get_neighbor(self.anion_a, self.cation))
                print("New alpha_LiLi:", self.alpha_LiLi(), "Li-Li:", self.get_neighbor_LiLi())
                if abs(self.alpha_LiLi() - target_alpha_LiLi) <= tol:
                    print("Final F-Li:", self.get_neighbor(self.anion_a, self.cation), "Final Li-Li:", self.get_neighbor_LiLi())
                    print("Target alpha_LiLi reached")
                    break
        else:
            print("Target alpha_LiLi not reached")

    def to_file(self, path):
        self.structure.to(filename=path)

for i in range(3):
    sro_gen_6 = SRO("TM4_10_POSCAR")
    sro_gen_6.run(200, target_alpha=-0.111, target_alpha_LiLi=-0.006, rate=100, tol=0.002, random_seed=i)
    if abs(sro_gen_6.alpha()+0.111) <0.002 and abs(sro_gen_6.alpha_LiLi()+0.006)<0.002:
        sro_gen_6.to_file("TM4_POSCAR")
        break