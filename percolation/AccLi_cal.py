"""
Dribble - Percolation Simulation on Lattices

Analyze the ionic percolation properties of an input structure.

"""

import argparse
import sys
import time

import numpy as np
import json

from dribble.io import Input
from dribble.percolator import Percolator
from dribble.lattice import Lattice
from dribble.misc import uprint


def write_to_file(out_filename, in_filename):
    content = {
        "structure": in_filename,
        "formula_units": 1,
        "sublattices": {
            "cations": {
                "description": "Cation sites",
                "sites": {"species": ["Li"]},
                "initial_occupancy": {"Li": 1.0}
            },
            "cation2": {
                "description": "Cation2 sites",
                "sites": {"species": ["Mn", "Ti"]},
                "initial_occupancy": {"TM": 1.0}
            },
            "oxygen": {
                "description": "Oxygen sites",
                "sites": {"species": ["O", "F"]},
                "ignore": True
            }
        },
        "bonds": [
            {
                "sublattices": ["cations", "cations"],
                "bond_rules": [
                    ["MinCommonNNNeighborsBR", {"num_neighbors": 2}]
                ]
            }
        ],
        "percolating_species": ["Li"],
        "flip_sequence": [["TM", "Li"]]
    }

    with open(out_filename, 'w') as outfile:
        json.dump(content, outfile, indent=4)


def check_if_percolating(percolator, inp, save_clusters, tortuosity):
    noccup = percolator.num_occupied
    nspan = percolator.check_spanning(verbose=False,
                                      save_clusters=save_clusters,
                                      static_sites=inp.static_sites)
    if (nspan > 0):
        uprint(" The initial structure is percolating.\n")
        uprint(" Fraction of accessible sites: {}\n".format(float(nspan) / float(noccup)))
        if tortuosity:
            for c in percolator.percolating_clusters:
                t_min, t_mean, t_std = percolator.get_tortuosity(c)
                uprint(" Tortuosity of cluster {} (min, mean): ".format(c)
                       + "{:5.3f}, {:5.3f} +/- {:5.3f}".format(
                    t_min, t_mean, t_std))
            uprint("")
    else:
        uprint(" The initial structure is NOT percolating.\n")
        uprint(" Fraction of accessible sites: 0.0\n")


def calc_critical_concentration(percolator, save_clusters, samples,
                                file_name, sequence):
    if save_clusters:
        (pc_site_any, pc_site_two, pc_site_all, pc_bond_any,
         pc_bond_two, pc_bond_all) = percolator.percolation_point(
            sequence, samples=samples, file_name=file_name + ".vasp")
    else:
        (pc_site_any, pc_site_two, pc_site_all, pc_bond_any,
         pc_bond_two, pc_bond_all) = percolator.percolation_point(
            sequence, samples=samples)

    uprint(" Critical site (bond) concentrations to find a "
           "wrapping cluster\n")

    uprint(" in one or more dimensions   p_c1 = {:.8f}  ({:.8f})".format(
        pc_site_any, pc_bond_any))
    uprint(" in two or three dimensions  p_c2 = {:.8f}  ({:.8f})".format(
        pc_site_two, pc_bond_two))
    uprint(" in all three dimensions     p_c3 = {:.8f}  ({:.8f})".format(
        pc_site_all, pc_bond_all))

    uprint("")


def calc_p_infinity(percolator, samples, save_raw, file_name, sequence):
    plist = np.arange(0.01, 1.00, 0.01)
    (Q, X) = percolator.calc_p_infinity(
        plist, sequence, samples=samples,
        save_discrete=save_raw)

    # integrate susceptibility X in order to normalize it
    intX = np.sum(X) * (plist[1] - plist[0])

    fname = file_name + ".infty"
    uprint(" Writing results to: {}\n".format(fname))

    with open(fname, 'w') as f:
        f.write("# {:^10s}   {:>10s}   {:>15s}   {:>15s}\n".format(
            "p", "P_infty(p)", "Chi(p)", "normalized"))
        for p in range(len(plist)):
            f.write("  {:10.8f}   {:10.8f}   {:15.8f}   {:15.8f}\n".format(
                plist[p], Q[p], X[p], X[p] / intX))


def calc_p_wrapping(percolator, samples, save_raw, file_name, sequence):
    plist = np.arange(0.01, 1.00, 0.01)
    (Q, Qc) = percolator.calc_p_wrapping(
        plist, sequence, samples=samples,
        save_discrete=save_raw)

    fname = file_name + ".wrap"
    uprint(" Writing results to: {}\n".format(fname))

    with open(fname, 'w') as f:
        f.write("# {:^10s}   {:>10s}   {:>10s}\n".format(
            "p", "P_wrap(p)", "cumulative"))
        for p in range(len(plist)):
            f.write("  {:10.8f}   {:10.8f}   {:10.8f}\n".format(
                plist[p], Q[p], Qc[p]))


def calc_inaccessible_sites(percolator, samples, save_raw, file_name,
                            sequence, species):
    plist = np.arange(0.01, 1.00, 0.01)
    (F_inacc, nclus) = percolator.inaccessible_sites(
        plist, sequence, species, samples=samples,
        save_discrete=save_raw)

    fname = file_name + ".inacc"
    uprint(" Writing results to: {}\n".format(fname))

    with open(fname, 'w') as f:
        f.write("# {:^10s}   {:>10s}   {:>10s}\n".format(
            "p", "F_inacc(p)", "N_percol(p)"))
        for p in range(len(plist)):
            f.write("  {:10.8f}   {:10.8f}   {:12.6f}\n".format(
                plist[p], F_inacc[p], nclus[p]))


def calc_mean_tortuosity(percolator, samples, file_name, sequence):
    F_tort = percolator.mean_tortuosity(
        sequence, samples=samples)

    fname = file_name + ".tortuosity"
    uprint(" Writing results to: {}\n".format(fname))

    with open(fname, 'w') as f:
        f.write("# {:^10s}  {:^10s}   {:s}\n".format(
            "N", "p", "Tortuosity(p)"))
        N = len(F_tort)
        for i, T in enumerate(F_tort):
            f.write("  {:10d}  {:10.8f}   {:10.8f}\n".format(
                i + 1, (i + 1) / float(N), T))


def compute_percolation(input_file, structure_file, samples,
                        save_clusters, save_raw, file_name, pc, check,
                        pinf, pwrap, inaccessible, tortuosity,
                        mean_tortuosity, supercell):
    if not (check or pc or pinf or pwrap or inaccessible or mean_tortuosity):
        print("\n Nothing to do.")
        print(" Please specify the quantity to be calculated.")
        print(" Use the `--help' flag to list all options.\n")
        sys.exit()

    input_params = {}
    #     if structure_file is not None:
    #         uprint("\n Reading structure from file: {}".format(structure_file))
    #         input_params['structure'] = structure_file

    #     uprint("\n Parsing input file '{}'...".format(input_file), end="")
    inp = Input.from_file(input_file, **input_params)
    #     uprint(" done.")

    #     uprint("\n Setting up lattice and neighbor lists...", end="")
    lattice = Lattice.from_input_object(inp, supercell=supercell)
    #     uprint(" done.")
    #     uprint(lattice)

    #     uprint(" Initializing percolator...", end="")
    percolator = Percolator.from_input_object(inp, lattice, verbose=True)
    #     uprint(" done.")

    #     uprint("\n MC percolation simulation\n -------------------------\n")

    if check:  # check, if initial structure is percolating
        check_if_percolating(percolator, inp, save_clusters, tortuosity)
    if pc:  # calculate critical site concentrations
        calc_critical_concentration(percolator, save_clusters, samples,
                                    file_name, inp.flip_sequence)
    if pinf:  # estimate P_infinity(p)
        calc_p_infinity(percolator, samples, save_raw, file_name,
                        inp.flip_sequence)
    if pwrap:  # estimate P_wrapping(p)
        calc_p_wrapping(percolator, samples, save_raw, file_name,
                        inp.flip_sequence)
    if inaccessible is not None:  # fraction of inaccessible sites
        calc_inaccessible_sites(percolator, samples, save_raw,
                                file_name, inp.flip_sequence,
                                inaccessible)
    if mean_tortuosity:  # tortuosity as function of concentration
        calc_mean_tortuosity(percolator, samples, file_name,
                             inp.flip_sequence)

    dt = time.gmtime(time.process_time())
#     uprint(" All done.  Elapsed CPU time: {:02d}h{:02d}m{:02d}s\n".format(
#             dt.tm_hour, dt.tm_min, dt.tm_sec))


ls = [0, 1, 2, 3, 4, 5, 6, 7,8,9]
tm2 = []
for i in ls:
    file_names = './FLi0/TM2_' + str(i) + '_POSCAR'
    tm2.append(file_names)

# 将标准输出重定向到文件
original_stdout = sys.stdout
sys.stdout = open("output.txt", "w")

for i in range(len(tm2)):
    write_to_file('input-bond-rule111.json', tm2[i])

    compute_percolation(input_file="input-bond-rule111.json",
                        structure_file=None,
                        samples=10,
                        save_clusters=None,
                        save_raw=None,
                        file_name=None,
                        pc=None,
                        check=True,
                        pinf=None,
                        pwrap=None,
                        inaccessible=None,
                        tortuosity=None,
                        mean_tortuosity=None,
                        supercell=(2, 2, 2))

# 恢复标准输出
sys.stdout = original_stdout

import re
# 打开文件以读取内容
with open("output.txt", "r") as file:
    # 读取文件内容
    file_content = file.read()
    # 使用正则表达式提取数字
    numbers = re.findall(r'\d+\.\d+', file_content)
    # 将提取的数字转换为浮点数，保留5位小数
    numbers = [round(float(num), 5) for num in numbers]
    # 打印或处理提取的数字列表
    print("Extracted numbers:", numbers)
