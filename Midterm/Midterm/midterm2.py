import numpy as np
import random
import os, subprocess
import copy
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from numpy import genfromtxt

it = 0;
def mid_quest_two(N, vc, delta, ep):

    for x in range(0, 5):
        nTest = (8 / (ep ** 2)) * np.log((4 * ((2 * N) ** vc) + 1) / delta)
        N = nTest
        print nTest
    return nTest


def main():

    x = mid_quest_two(1000000, 10, .05, .05)
    print "answer after 5 " + str(x)
main()
