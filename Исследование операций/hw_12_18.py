#!/usr/bin/env python3

import sys
import copy
import itertools

import logging  # logging

import pprint   # nice dictionary output
pp = pprint.PrettyPrinter(indent=4)

import numpy as np

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler() # stderr by default
formatter = logging.Formatter(    # specify output format
    fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S %z')
handler.setFormatter(formatter)
logger.addHandler(handler)

def main():
    B = np.array([[float(val) for val in line.split()] for line in sys.stdin.readlines()])
    print(f"B:\n{B!s}\n\n")
    D = np.log2(B)
    print(f"D:\n{D!s}\n\n")
    step_matrixes = []
    for step in range(len(B)):
        step_matrix = copy.deepcopy(D)
        for i, j in itertools.product(range(len(B)),range(len(B))):
            step_matrix[i, j] = D[i, step] + D[step, j]
        step_matrixes.append(step_matrix)
        print(f"D^{step+1}:\n{step_matrix!s}")
    print('\n\n')

    Dwave = np.zeros_like(D)
    for matrix in step_matrixes:
        Dwave += (matrix / len(B))
    print(f"D^wave:\n{Dwave!s}\n\n")

    Bwave = 2**Dwave
    print(f"B^wave:\n{Bwave!s}\n\n")

    # дальше я уже не уверен
    x0 = np.array([Bwave[i,i] for i in range(len(Bwave))])
    x = x0 @ Bwave
    print("x=Bwave * x0")
    print("x0:\n{x0!s}")
    print("x:\n{x!s}")
    print("мин знач:", min(x))
    print("макс знач:", max(x))



if __name__ == "__main__":
    main()
