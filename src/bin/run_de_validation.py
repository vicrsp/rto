import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# add the parent folder to path
lib_path = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(lib_path)

from optimization.de import DifferentialEvolution

# create a test function
def f_test(x):
    fx = (x[0] - 3) ** 2 + (x[1] - 2) ** 2
    g = x[0] - x[1]  # x2 > x1

    return fx, g

def multi_f(x):
	## Himmel Blau function!
    #f(3.0,2.0)=0.0
    #f(-2.805118,3.131312)=0.0
    #f(-3.779310,-3.283186)=0.0
    #f(3.584428,-1.848126)=0.0
	sum_ = pow((pow(x[0],2) + x[1] - 11),2) + pow((pow(x[1],2) + x[0] - 7),2)
	g = 26.0 - pow((x[0]-5.0), 2) - pow(x[1],2)#constraints.

	return sum_, g

de = DifferentialEvolution(multi_f, [-10, -10], [10, 10], max_generations=500)
de.run()
