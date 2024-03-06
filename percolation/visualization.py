from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.structure import Structure, Element
import numpy as np

cnn = CrystalNN()

# 对象结构tm2和四面体中心文件tet_site

tm2 = Structure.from_file("TM2_0_POSCAR1")
tet_site = Structure.from_file("tet_site2000.vasp")
tm3 = Structure.from_file("TM2_0_POSCAR1")

# 将对象结构的阴离子删去
for i in range(1999, 999, -1):
    tm2.pop(i)
print(tm2.num_sites)

# 将四面体位点加入对象结构中，四面体位点符号都是“F”
tet_list = []
for i in range(0, 2000):
    tet_list.append(tet_site[i])

for i in range(len(tet_list)):
    tm2.append(tet_list[i].specie, tet_list[i].frac_coords)
print(tm2.num_sites)

site_num = 0
site_list = []

# for i in range(1, 100) 的作用是创建一个循环，迭代变量 i 从1递增到99（不包括100）
# 四面体位点符号都是“F”
for i in range(1000,3000):
    if tm2.species[i] == Element("F"):
        if cnn.get_cn_dict(tm2, i).get("Li", 0) == 4:
            site_list.append(i)
            site_num=site_num+1
# site_num是0-TM四面体中心数量
print(site_num)

# 统计0-TM处Li的含量，会重复统计，没有太大物理意义，，尽管将重复的Li剔除得到的结果也不行，主要看可渗流Li含量
li_list=[]
for i in site_list:
    for j in range(4):
        activeLi = cnn.get_nn(tm2, i)[j].index
        li_list.append(activeLi)
print(len(li_list))
# list(set(li_list)) 的作用是将一个列表 li_list 中的重复元素去除，返回一个新的列表，其中仅包含唯一的元素
num_0TMLi = list(set(li_list))


for i in [num_0TMLi]:
    tm3[i] = "C"

for i in range(1999, -1,-1):
    if tm3.species[i] != Element("C"):
        tm3.pop(i)

tm3.to(filename="0tm_li.vasp",fmt="POSCAR")