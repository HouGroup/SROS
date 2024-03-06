"""
Both_FLi_LiLi.py

This module provides functionality for swapping cations in a crystal structure to achieve specified alpha values for FLi and LiLi.

The module follows a two-step process:
1. Achieve the target alpha value for FLi while performing cation swaps.
2. While maintaining the achieved FLi alpha value, continue swapping cations to reach the specified alpha value for LiLi.

Functions:
    - swap_cations_both(structure, target_alpha_FLi, target_alpha_LiLi): Performs cation swaps to achieve target alpha values for FLi and LiLi.
    - additional_function(): Any additional helper function relevant to the cation swapping process.

Example Usage:

    from mypackage import Both_FLi_LiLi
    both_flilili = Both_FLi_LiLi.SRO("D:\\Users\\ASUS\\Desktop\\Computational Practice\\10.27\\TM_FLI-0.325-LLILI0.083\\TM2\\TM2_1_POSCAR")
    both_flilili.run(50, target_alpha=0, target_alpha_LiLi=0, rate=100, tol=0.02, random_seed=2)

Note:
    - This module assumes that the crystal structure is provided in a suitable format compatible with the cation swapping algorithm.
    - The two-step process ensures that the target alpha values for FLi and LiLi are independently achieved.
    - The effectiveness of achieving the exact target alphas may depend on various factors, and the results should be validated.

For more information on the theory and methods used, refer to the relevant literature or documentation.

Author: Liaojh
Date: January 23, 2024
"""

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
            # 第二近邻为阳离子-阳离子配位数为12
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
            # 第二近邻为阳离子-阳离子配位数为12
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
            if not target:  # 如果 target 为 False
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
            if target:  # 如果 target 为 True
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
        c_site = c_idxs[random.randrange(len(c_idxs))]  # c_site是任意一个Li的位置
        d_site = d_idxs[random.randrange(len(d_idxs))]  # d_site是任意一个TM的位置

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
                self.a_LiLi = self.alpha_LiLi_fix()
                print("Cannot find correct cations")
                return self.a_LiLi

        old_alpha_LiLi = self.a_LiLi
        if random.random() < prob:  # 结构Li周围的Li比目标值少
            if not target:  # 这里对应的是target=False，即Li周围选的不是Li，而TM周围选的是Li
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
        else:  # 结构Li周围的Li比目标值多
            if target:  # 这里对应的是target=True，即Li周围选的是Li，而TM周围选的不是Li
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

    def run(self, max_steps: int,
            target_alpha: Union[int, float] = 0,
            target_alpha_LiLi: Union[int, float] = 0,
            rate: Union[int, float] = 1,
            tol: Union[int, float] = 0.05,
            random_seed: Union[None, int] = None,
            ):
        """
        主函数，执行任务
        """
        random.seed(random_seed)
        print("Innitial alpha:", self.a, "Innitial alphaLiLi:", self.a_LiLi)
        for i in range(max_steps):
            if abs(self.a - target_alpha) <= tol:
                print("Target alpha reached")
                break
            self.exchange(target_alpha, rate, random_seed=random.randrange(1000))
            print("步数：", i, "New alpha:", self.a)

        self.a_LiLi, self.a_LiLi_dict = self.alpha_LiLi()
        if abs(self.alpha_fix() - target_alpha) <= tol:
            for i in range(max_steps):
                if abs(self.a_LiLi - target_alpha_LiLi) <= tol:
                    print("Target alpha_LiLi reached")
                    break
                self.exchange_LiLi(target_alpha_LiLi, target_alpha, tol, rate, random_seed=random.randrange(1000))
                print("步数：", i, "New alpha_LiLi:", self.a_LiLi)
        else:
            print("Target alpha not reached")

        if abs(self.alpha_LiLi_fix() - target_alpha_LiLi) <= tol:
            print("Both target alpha and alpha_LiLi reached.")
        else:
            print("Target alpha_LiLi not reached")

    def to_file(self, path):
        self.structure.to(filename=path)
