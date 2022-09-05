#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:31:02 2022

@author: jiangyunhui
"""
import numpy as np

import J_5
from J_5 import data_generation_all
from J_5 import MMAlgorithm_all
from data_generation_all import data_generation
from MMAlgorithm_all import MMAlgorithm

# test
#cumulative link model
data=data_generation(size=20,num_game=10,dynamic_range=2, sparsity=1, theta_1=2, theta_2=4)
graph_5, graph_4, graph_3, graph_2, graph_1, gamma = data.generate_five_data_clm()
alg=MMAlgorithm(size=20, graph_5=graph_5, graph_4=graph_4, graph_3=graph_3, graph_2=graph_2, graph_1=graph_1)
alg.clm_5_given_theta(theta_1=2,theta_2=4)
alg.clm_5(iteration=20)

#adjacent categories logit model
data=data_generation(size=100,num_game=10,dynamic_range=2, sparsity=1, theta_1=10, theta_2=20)
graph_5, graph_4, graph_3, graph_2, graph_1, gamma = data.generate_five_data_aclm()
alg=MMAlgorithm(size=100, graph_5=graph_5, graph_4=graph_4, graph_3=graph_3, graph_2=graph_2, graph_1=graph_1)

alg.aclm_5_given_theta(theta_1=9,theta_2=9)

alg.aclm_5(iteration=20)


