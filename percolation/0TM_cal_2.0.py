from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.structure import Structure, Element
import numpy as np

# 统计四面体中心位点最近邻原子时要用CrystalNN，不能用BrunnerNN_real会出错
cnn = CrystalNN()

tet_site = Structure.from_file("tet_site2000.vasp")
anion_list = []
for i in range(0,2000):
    anion_list.append(tet_site[i])

# 统计一系列初始结构的0-TM数目情况
ls = [0,1,2,3,4,5,6,8,9]

print("**********************FLi0***********************")
tm2 = []
ls_0tm = []
for i in ls:
    name = Structure.from_file('./FLi0/TM2_' + str(i) + '_POSCAR')
    tm2.append(name)
for i in range(len(tm2)):
    for j in range(1999,999,-1):# 后一半都是阴离子，删去
        tm2[i].pop(j)
    for k in range(len(anion_list)):
        tm2[i].append(anion_list[k].specie, anion_list[k].frac_coords) # 加入四面体中心位点 F 便于统计
    site0tm_num = 0
    for n in range(1000,2000): # 对四面体中心位点 F 进行统计
        if tm2[i].species[n] == Element("F"):
            if cnn.get_cn_dict(tm2[i], n).get("Li", 0) == 4:
                site0tm_num = site0tm_num + 1
    ls_0tm.append(site0tm_num)
print(ls_0tm)

print("**********************FLi-0.05***********************")
tm2 = []
ls_0tm = []
for i in ls:
    name = Structure.from_file('./FLi-0.05/TM2_' + str(i) + '_POSCAR')
    tm2.append(name)
for i in range(len(tm2)):
    for j in range(1999,999,-1):# 后一半都是阴离子，删去
        tm2[i].pop(j)
    for k in range(len(anion_list)):
        tm2[i].append(anion_list[k].specie, anion_list[k].frac_coords) # 加入四面体中心位点 F 便于统计
    site0tm_num = 0
    for n in range(1000,2000): # 对四面体中心位点 F 进行统计
        if tm2[i].species[n] == Element("F"):
            if cnn.get_cn_dict(tm2[i], n).get("Li", 0) == 4:
                site0tm_num = site0tm_num + 1
    ls_0tm.append(site0tm_num)
print(ls_0tm)


print("**********************FLi-0.1***********************")
tm2 = []
ls_0tm = []
for i in ls:
    name = Structure.from_file('./FLi-0.1/TM2_' + str(i) + '_POSCAR')
    tm2.append(name)
for i in range(len(tm2)):
    for j in range(1999,999,-1):# 后一半都是阴离子，删去
        tm2[i].pop(j)
    for k in range(len(anion_list)):
        tm2[i].append(anion_list[k].specie, anion_list[k].frac_coords) # 加入四面体中心位点 F 便于统计
    site0tm_num = 0
    for n in range(1000,2000): # 对四面体中心位点 F 进行统计
        if tm2[i].species[n] == Element("F"):
            if cnn.get_cn_dict(tm2[i], n).get("Li", 0) == 4:
                site0tm_num = site0tm_num + 1
    ls_0tm.append(site0tm_num)
print(ls_0tm)


print("**********************FLi-0.15***********************")
tm2 = []
ls_0tm = []
for i in ls:
    name = Structure.from_file('./FLi-0.15/TM2_' + str(i) + '_POSCAR')
    tm2.append(name)
for i in range(len(tm2)):
    for j in range(1999,999,-1):# 后一半都是阴离子，删去
        tm2[i].pop(j)
    for k in range(len(anion_list)):
        tm2[i].append(anion_list[k].specie, anion_list[k].frac_coords) # 加入四面体中心位点 F 便于统计
    site0tm_num = 0
    for n in range(1000,2000): # 对四面体中心位点 F 进行统计
        if tm2[i].species[n] == Element("F"):
            if cnn.get_cn_dict(tm2[i], n).get("Li", 0) == 4:
                site0tm_num = site0tm_num + 1
    ls_0tm.append(site0tm_num)
print(ls_0tm)


print("**********************FLi-0.2***********************")
tm2 = []
ls_0tm = []
for i in ls:
    name = Structure.from_file('./FLi-0.2/TM2_' + str(i) + '_POSCAR')
    tm2.append(name)
for i in range(len(tm2)):
    for j in range(1999,999,-1):# 后一半都是阴离子，删去
        tm2[i].pop(j)
    for k in range(len(anion_list)):
        tm2[i].append(anion_list[k].specie, anion_list[k].frac_coords) # 加入四面体中心位点 F 便于统计
    site0tm_num = 0
    for n in range(1000,2000): # 对四面体中心位点 F 进行统计
        if tm2[i].species[n] == Element("F"):
            if cnn.get_cn_dict(tm2[i], n).get("Li", 0) == 4:
                site0tm_num = site0tm_num + 1
    ls_0tm.append(site0tm_num)
print(ls_0tm)


print("**********************FLi-0.25***********************")
tm2 = []
ls_0tm = []
for i in ls:
    name = Structure.from_file('./FLi-0.25/TM2_' + str(i) + '_POSCAR')
    tm2.append(name)
for i in range(len(tm2)):
    for j in range(1999,999,-1):# 后一半都是阴离子，删去
        tm2[i].pop(j)
    for k in range(len(anion_list)):
        tm2[i].append(anion_list[k].specie, anion_list[k].frac_coords) # 加入四面体中心位点 F 便于统计
    site0tm_num = 0
    for n in range(1000,2000): # 对四面体中心位点 F 进行统计
        if tm2[i].species[n] == Element("F"):
            if cnn.get_cn_dict(tm2[i], n).get("Li", 0) == 4:
                site0tm_num = site0tm_num + 1
    ls_0tm.append(site0tm_num)
print(ls_0tm)


print("**********************FLi-0.3***********************")
tm2 = []
ls_0tm = []
for i in ls:
    name = Structure.from_file('./FLi-0.3/TM2_' + str(i) + '_POSCAR')
    tm2.append(name)
for i in range(len(tm2)):
    for j in range(1999,999,-1):# 后一半都是阴离子，删去
        tm2[i].pop(j)
    for k in range(len(anion_list)):
        tm2[i].append(anion_list[k].specie, anion_list[k].frac_coords) # 加入四面体中心位点 F 便于统计
    site0tm_num = 0
    for n in range(1000,2000): # 对四面体中心位点 F 进行统计
        if tm2[i].species[n] == Element("F"):
            if cnn.get_cn_dict(tm2[i], n).get("Li", 0) == 4:
                site0tm_num = site0tm_num + 1
    ls_0tm.append(site0tm_num)
print(ls_0tm)