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

    def concen(self, tm: str, element: str):
        if tm == "tm2":
            if element == "Mn":
                c_element = 8 / 40
            elif element == "Ti":
                c_element = 6 / 40
            elif element == "Li":
                c_element = 26 / 40
            elif element == "F":
                c_element = 6 / 40
            elif element == "O":
                c_element = 34 / 40
        elif tm == "tm4":
            if element == "Mn":
                c_element = 8 / 40
            elif element == "Ti":
                c_element = 2 / 40
            elif element == "Nb":
                c_element = 4 / 40
            elif element == "Li":
                c_element = 26 / 40
            elif element == "F":
                c_element = 6 / 40
            elif element == "O":
                c_element = 34 / 40
        elif tm == "tm6":
            if element == "Mn":
                c_element = 4 / 40
            elif element == "Co":
                c_element = 2 / 40
            elif element == "Cr":
                c_element = 2 / 40
            elif element == "Ti":
                c_element = 2 / 40
            elif element == "Nb":
                c_element = 4 / 40
            elif element == "Li":
                c_element = 26 / 40
            elif element == "F":
                c_element = 6 / 40
            elif element == "O":
                c_element = 34 / 40
        return c_element

    def alpha_cation(self, tm: str, cationA: str, cationB: str):
        """
        计算阳离子WC SRO值，以A原子为中心,计算周围B原子
        """
        structrue_dup = self.structure.copy()
        last_idx = int(self.structure.num_sites - 1)
        mid_idx = int(self.structure.num_sites / 2 - 1)
        for i in range(last_idx, mid_idx, -1):
            structrue_dup.pop(i)

        cationA_idxs = self.get_idxs(cationA)
        alpha_list = []
        for i in cationA_idxs:
            # P is the probability of finding second nearest neighbor cation Li adjacent to cation Li.
            # 第二近邻为阳离子-阳离子配位数为12
            P = self.bnn.get_cn_dict(structrue_dup, i).get(cationB, 0) / self.cation_cn
            alpha_list.append(1 - P / self.concen(tm, cationB))

        return np.mean(alpha_list)

    def alpha_anion_cation(self, tm: str, anionA: str, cationA: str):
        """
        Calculate alpha where anion is the i species, cation is the j species, as defined in PNAS, 2021, 118, e2020540118.
        """
        if anionA == "F":
            anion_idxs = self.get_idxs(self.anion_a)
        elif anionA == "O":
            anion_idxs = self.get_idxs(self.anion_b)
        alpha_list = []
        for i in anion_idxs:
            # P is the probability of finding cation Li adjacent to anion F
            P = self.cnn.get_cn_dict(self.structure, i).get(cationA, 0) / self.anion_cn
            alpha_list.append(1 - P / self.concen(tm, cationA))
        return np.mean(alpha_list)

    def alpha_anion_anion(self, tm: str, anionA: str, anionB: str):
        """
        Calculate alpha where anion is the i species, anionB is the j species, as defined in PNAS, 2021, 118, e2020540118.
        """
        structrue_dup = self.structure.copy()
        mid_idx = int(self.structure.num_sites / 2)
        for i in range(mid_idx):
            structrue_dup.pop(0)

        anion_idxs = []
        for i in range(structrue_dup.num_sites):
            if structrue_dup.species[i] == Element(anionA):
                anion_idxs.append(i)

        alpha_list = []
        for i in anion_idxs:
            P = self.bnn.get_cn_dict(structrue_dup, i).get(anionB, 0) / 12
            alpha_list.append(1 - P / self.concen(tm, anionB))

        return np.mean(alpha_list)

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
        该函数用于显示指定阳离子（idx）周边12个阳离子的信息
        """
        structrue_dup = self.structure.copy()
        last_idx = int(self.structure.num_sites - 1)
        mid_idx = int(self.structure.num_sites / 2 - 1)
        for i in range(last_idx, mid_idx, -1):
            structrue_dup.pop(i)
        env = self.bnn.get_nn(structrue_dup, idx)

        return env

    def to_file(self, path):
        self.structure.to(filename=path)


def run(self):
    tm4 = SRO(r"D:\Users\ASUS\Desktop\Computational Practice\0103\Hackathon\TM4_1_POSCAR")

    print("*************************************************")
    print("以下是最近邻的阴离子-阳离子的熵")

    lsf = []
    lsf.append(tm4.alpha())
    lsf.append(tm4.alpha_anion_cation("tm4", "F", "Mn"))
    lsf.append(tm4.alpha_anion_cation("tm4", "F", "Ti"))
    lsf.append(tm4.alpha_anion_cation("tm4", "F", "Nb"))
    lsf.append(tm4.alpha_anion_cation("tm4", "O", "Li"))
    lsf.append(tm4.alpha_anion_cation("tm4", "O", "Mn"))
    lsf.append(tm4.alpha_anion_cation("tm4", "O", "Ti"))
    lsf.append(tm4.alpha_anion_cation("tm4", "O", "Nb"))

    print(lsf)
    lsx1x2 = [0.0975, 0.03, 0.0075, 0.015, 0.5525, 0.17, 0.0425, 0.085]
    tot = 0
    for i in range(len(lsf)):
        if (1 - lsf[i]) == 0:
            tot = tot + 0
        else:
            # 处理不合法的情况，例如给 tot 赋一个默认值
            tot = tot + lsx1x2[i] * (1 - lsf[i]) * math.log((1 - lsf[i]), math.e)
    print(tot * 6)


    print("*************************************************")
    print("以下是阳离子-阳离子的熵")

    lsli = []
    lsli.append(tm4.alpha_LiLi())
    lsli.append(tm4.alpha_cation("tm4","Li","Mn"))
    lsli.append(tm4.alpha_cation("tm4","Li","Ti"))
    lsli.append(tm4.alpha_cation("tm4","Li","Nb"))
    lsli.append(tm4.alpha_cation("tm4","Li","Mn"))
    lsli.append(tm4.alpha_cation("tm4","Mn","Mn"))
    lsli.append(tm4.alpha_cation("tm4","Mn","Ti"))
    lsli.append(tm4.alpha_cation("tm4","Mn","Nb"))
    lsli.append(tm4.alpha_cation("tm4","Li","Ti"))
    lsli.append(tm4.alpha_cation("tm4","Mn","Ti"))
    lsli.append(tm4.alpha_cation("tm4","Ti","Ti"))
    lsli.append(tm4.alpha_cation("tm4","Ti","Nb"))
    lsli.append(tm4.alpha_cation("tm4","Li","Nb"))
    lsli.append(tm4.alpha_cation("tm4","Mn","Nb"))
    lsli.append(tm4.alpha_cation("tm4","Ti","Nb"))
    lsli.append(tm4.alpha_cation("tm4","Nb","Nb"))

    print(lsli)
    lsx1x2 = [0.4225, 0.13, 0.0325, 0.065, 0.13, 0.04, 0.01, 0.02, 0.0325, 0.01, 0.0025, 0.005, 0.065, 0.02, 0.005, 0.01]
    tot = 0
    for i in range(len(lsli)):
        if (1 - lsli[i]) == 0:
            tot = tot + 0
        else:
            # 处理不合法的情况，例如给 tot 赋一个默认值
            tot = tot + lsx1x2[i] * (1 - lsli[i]) * math.log((1 - lsli[i]), math.e)
    print(tot * 6)


    print("*************************************************")
    print("以下是阴离子-阴离子的熵")

    lsff = []
    lsff.append(tm4.alpha_anion_anion("tm4","F","F"))
    lsff.append(tm4.alpha_anion_anion("tm4","F","O"))
    lsff.append(tm4.alpha_anion_anion("tm4","F","O"))
    lsff.append(tm4.alpha_anion_anion("tm4","O","O"))

    print(lsff)
    lsx1x2 = [0.0225, 0.1275, 0.1275, 0.7225]
    tot = 0
    for i in range(len(lsff)):
        if (1 - lsff[i]) == 0:
            tot = tot + 0
        else:
            # 处理不合法的情况，例如给 tot 赋一个默认值
            tot = tot + lsx1x2[i] * (1 - lsff[i]) * math.log((1 - lsff[i]), math.e)


