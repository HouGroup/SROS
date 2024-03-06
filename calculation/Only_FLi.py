# 代码取自D:\Python\Jupyter\MC-try_better\FLi-test4.ipynb

"""
Only_FLi.py

This module provides functionality for swapping cations in a crystal structure to achieve a specified alpha value for FLi.

The primary purpose of this module is to manipulate crystal structures by exchanging positively charged ions (cations) in a way that influences the SRO, ultimately leading to a targeted alpha value.

Functions:
    - swap_cations(structure, target_alpha): Performs cation swaps in the given crystal structure to achieve the desired alpha value.
    - additional_function(): Any additional helper function relevant to the cation swapping process.

Example Usage:
    from mypackage import Only_FLi

    onlyfli = Only_FLi.SRO("D:\\Users\\ASUS\\Desktop\\Computational Practice\\10.27\\TM_FLI-0.325-LLILI0.083\\TM2\\TM2_1_POSCAR")

    onlyfli.run(50, target_alpha=0, rate=100, tol=0.01, random_seed=1)

Note:
    - This module assumes that the crystal structure is provided in a suitable format compatible with the cation swapping algorithm.
    - The calculations are specific to FLi, and adjustments may be needed for other crystal structures.
    - The effectiveness of achieving the exact target alpha may depend on various factors, and the result should be validated.

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
        # self.all_idxs = list(range(0, self.structure.num_sites)) #仅考虑FLi时未用到，注释以提高效率
        self.cnn = CrystalNN()
        self.bnn = BrunnerNN_real()
        self.anion_cn = 6  # Anion coordination number, which is 6 in DRX
        self.cation_cn = 12  # cation-cation coordination number, which is 12 in DRX.
        self.a, self.a_dict = self.alpha()
        # self.a_LiLi = self.alpha_LiLi()  #仅考虑FLi时未用到，注释以提高效率

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
        # anion_idxs = self.get_idxs(self.anion_a)
        anion_idxs = self.a_idxs
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
        print("Changed new_alpha:", new_alpha)
        return new_alpha, new_dict

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
            # 尝试获取位置 a 处元素的元素名的字符串表示，如‘Li’，如果有 to_pretty_string 方法则使用它，否则使用 name 属性
            original_a = self.structure.species[a].to_pretty_string()
        except AttributeError:
            original_a = self.structure.species[a].name
        try:
            original_b = self.structure.species[b].to_pretty_string()
        except AttributeError:
            original_b = self.structure.species[b].name
        # 将位置 b 处的元素替换为位置 a 处的元素，将位置 a 处的元素替换为位置 b 处的元素
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

    def run(self, max_steps: int,
            target_alpha: Union[int, float] = 0,
            rate: Union[int, float] = 1,
            tol: Union[int, float] = 0.05,
            random_seed: Union[None, int] = None,
            ):
        """
        主函数，执行任务
        """
        random.seed(random_seed)
        print("Innitial alpha:", self.a, "Innitial F-Li:", self.get_neighbor(self.anion_a, self.cation))
        for i in range(max_steps):
            if abs(self.a - target_alpha) <= tol:
                print("Final F-Li:", self.get_neighbor(self.anion_a, self.cation))
                print("Target alpha reached.........")
                break
            self.exchange(target_alpha, rate, random_seed=random.randrange(1000))
            print("New alpha:", self.a)

    def to_file(self, path):
        self.structure.to(filename=path)
