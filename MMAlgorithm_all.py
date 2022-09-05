#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 14:27:59 2022

@author: jiangyunhui
"""
import pandas as pd
import numpy as np
import copy

class MMAlgorithm:
    
    def __init__(self, size, **graphs):
        
        """
        Parameters
        ----------
        size : The  number of players involved in the comparison.
        
        **graphs : The graphs for game outcomes, input must be of the form as:
            
            when J = 5
                graph_1 = graph_1, graph_2 = graph_2, graph_3 = graph_3, graph_4 = graph_4, graph_5 = graph_5;
            when J = 4
                graph_1 = graph_1, graph_2 = graph_2, graph_3 = graph_3, graph_4 = graph_4;
            when J = 3:
                win_graph = win_graph, tie_graph = tie_graph;
            when J = 2:
                win_graph = win_graph

        Returns
        -------
        check assumptions: index;
        algorithms: estimation(gamma), *estimation_theta, *num_iter

        """
        self.size = size
        
        for key, value in graphs.items():
            self.__dict__[key] = value
            
        self.matrix_1 = np.ones((self.size,self.size))
        self.matrix_2 = np.zeros((self.size,self.size))
    
        for i in range(self.size):
            for j in range(self.size):
                if i > j:
                    self.matrix_1[i,j], self.matrix_2[i,j] = 0,1
                elif i == j:
                    self.matrix_1[i,j], self.matrix_2[i,j] = 0,0

            
    def check_assumption_3_no_tie(self):  
        
        """
        check Assumption 3 only for J=2 cases
        """

        for i in range(self.size): # change 1000 to num_player
            win = np.sum(self.win_graph,axis=1)
            lose = np.sum(self.win_graph, axis = 0)
            outlier = np.union1d(np.where(win==0)[0], np.where(lose==0)[0]) 
            index = np.setdiff1d(np.array((range(self.size))),outlier)
            self.win_graph = (self.win_graph[:,index])[index,:]
            self.size = len(index)
            if outlier.size == 0:
                break
            
        return index
            
    def check_assumption_3(self):
        
        """
        check Assumption 3 only for J=3 cases
        """
        
        for i in range(self.size):
            
            win = np.sum(self.win_graph,axis=1)
            lose = np.sum(self.win_graph, axis = 0)
            tie = np.sum(self.tie_graph, axis = 1)
            total_game = win+tie+lose
            outlier = np.where(total_game==0)[0]
            index = np.setdiff1d(np.array((range(self.size))),outlier)
            self.win_graph = (self.win_graph[:,index])[index,:]
            self.tie_graph = (self.tie_graph[:,index])[index,:]
            self.size = len(index)
            if outlier.size == 0:
                break
        
        return index
    
    def check_assumption_1(self):
        
        """
        check Assumption 1 only for J=3 cases
        """
        
        num_player=self.size
        win_graph=self.win_graph.copy()
        for i in range(num_player):
            win_graph[i,i]=100
        
        outlier,outlier_2 = np.where(win_graph==0)
        outlier=np.unique(outlier)
#        outlier_2=np.unique(outlier_2)
#        outlier = np.union1d(outlier_1,outlier_2)
        index = np.setdiff1d(np.array((range(self.size))),outlier) 
        self.win_graph = (self.win_graph[:,index])[index,:]
        self.tie_graph = (self.tie_graph[:,index])[index,:]
        self.size = len(index)
 
        return index
    
    def mmAlgorithm_bt(self,iteration=1000,error=1e-5):

        
        # from win_graph to estimated vector
        # implement mm algorithm
        
        #initial = np.ones(num_player)
        #iteration = 1000
        win_graph=self.win_graph
        num_player=self.size
        win = np.sum(win_graph,axis=1)
        num_comparison_graph = np.transpose(win_graph) + win_graph
        

        initial=np.ones(num_player) # may set to be able to be self-defined later 
        previous=initial
        for K in range(iteration):
            last = previous.copy()
            prob_ij_matrix = np.zeros((num_player,num_player))
            for i in range(num_player):
                prob_ij_matrix[i] = previous[i] + previous  # Some correction
            
            g_matrix = num_comparison_graph/(prob_ij_matrix)   # Some correction
            g_matrix[np.diag_indices_from(g_matrix)] = 0 # since i != j
            g_matrix_sum = np.sum(g_matrix,axis=1)
            
            previous = (win)/g_matrix_sum
            previous = previous/previous[0]        # choose one as the benchmark
            
            if np.max(np.abs(last - previous)) < error:  # Converage criterion
                print('Pairwise Converge')
                estimation = previous
                print(estimation)
                break
        # return the optimal gamma estimated by mmAlgorithm_0
        return estimation
    
    def Davidson(self,iteration=1000,error=1e-9):
        
        win_graph=self.win_graph
        tie_graph=self.tie_graph
        num_player=self.size
        
        # compute MLE
        gamma=np.ones(num_player)
        theta=1
        initial = gamma.copy()
        initial_theta=theta
       
        win=np.sum(win_graph,axis=1)
        tie=np.sum(tie_graph,axis=1)
       
        comparison_graph = win_graph+np.transpose(win_graph)+tie_graph # w_ij + w_ji + t_ij
        
#       a = 'True'

        for K in range(iteration):
            last = initial.copy()
            last_theta=initial_theta
            
            # update gamma
            
            current_probability = np.zeros((num_player,num_player))
            for i in range(num_player):
                current_probability[i] = initial[i]/initial       # ij term = gamma_i/gamma_j
    
            transpose_current_probability = np.transpose(current_probability)   # ij term = gamma_j/gamma_i
            
            # compute the g_matrix
            up_matrix = (2 + last_theta*np.sqrt(transpose_current_probability)) * comparison_graph
            down_matrix = (current_probability + 1 + last_theta*np.sqrt(current_probability)) * initial  # It is true!
#            print('down_matrix')
#            print(down_matrix)
            g_matrix = up_matrix/down_matrix
            
            g_matrix_sum = np.sum(g_matrix,axis=1)
            
            initial = (2*win + tie)/g_matrix_sum # the numerator of gamma_matrix, multiply tie by 2
            initial = initial/np.sum(initial)
            
            # update theta
            T=np.sum(tie_graph)/2 # total num of ties
            
            for i in range(num_player):
                current_probability[i] = initial[i]/initial       # ij term = gamma_i/gamma_j
                
            up_matrix_theta = (2*win_graph+tie_graph)*(np.sqrt(current_probability)) 
            down_matrix_theta = (current_probability + 1 + last_theta*np.sqrt(current_probability))
            matrix_theta = up_matrix_theta/down_matrix_theta
            
            initial_theta=2*T/(np.sum(matrix_theta))
    
            if np.max(np.abs(last - initial))/np.max(np.abs(gamma)) < error and np.abs(last_theta-initial_theta) < error:
#                print(np.max(np.abs(last - initial))/np.max(np.abs(gamma)))
#                print(np.abs(last_theta-initial_theta))
                print('Converge')
                estimation = initial
                print(estimation)
                estimation_theta=initial_theta
                break
         
        estimation = initial
        estimation_theta=initial_theta
  
        return estimation, estimation_theta
           
    
    def Davidson_given_theta(self,iteration=1000,error=1e-9,theta=1):
        
        win_graph=self.win_graph
        tie_graph=self.tie_graph
        num_player=self.size
        
        # compute MLE
        gamma=np.ones(num_player)
        initial = gamma.copy()
       
        win=np.sum(win_graph,axis=1)
        tie=np.sum(tie_graph,axis=1)
       
        comparison_graph = win_graph+np.transpose(win_graph)+tie_graph # w_ij + w_ji + t_ij
        
#       a = 'True'

        for K in range(iteration):
            last = initial.copy()
            
            
            # update gamma
            
            current_probability = np.zeros((num_player,num_player))
            for i in range(num_player):
                current_probability[i] = initial[i]/initial       # ij term = gamma_i/gamma_j
    
            transpose_current_probability = np.transpose(current_probability)   # ij term = gamma_j/gamma_i
            
            # compute the g_matrix
            up_matrix = (2 + theta*np.sqrt(transpose_current_probability)) * comparison_graph
            down_matrix = (current_probability + 1 + theta*np.sqrt(current_probability)) * initial  # It is true!
            g_matrix = up_matrix/down_matrix
            
            g_matrix_sum = np.sum(g_matrix,axis=1)
            
            initial = (2*win + tie)/g_matrix_sum # the numerator of gamma_matrix, multiply tie by 2
            initial = initial/np.sum(initial)
            
            if np.max(np.abs(last - initial))/np.max(np.abs(gamma)) < error:
                print(np.max(np.abs(last - initial))/np.max(np.abs(gamma)))
                print('Converge')
                estimation = initial
                
                break
            
            
        estimation = initial
      
        return estimation
        
    def Rao_Kupper(self,iteration=1000,error=1e-9):
        
        win_graph=self.win_graph
        tie_graph=self.tie_graph
        num_player=self.size
        
        gamma=np.ones(num_player)/num_player
        theta=1
        initial = gamma.copy()
        initial_theta=theta
        
        win=np.sum(win_graph,axis=1)
        tie=np.sum(tie_graph,axis=1)

        si=win+tie # s_i
        sij=win_graph+tie_graph
        T=np.sum(tie)/2
        
        for k in range(iteration):
            
            last = initial.copy()
            last_theta=initial_theta
            # update gamma
            # down matrix_1
            down_matrix_1=np.zeros((num_player,num_player))
            for i in range(num_player):
                down_matrix_1[i]=initial[i]+initial_theta*initial
                
            down_matrix_2=np.zeros((num_player,num_player))
            for i in range(num_player):
                down_matrix_2[i]=initial_theta*initial[i]+initial
                 
            denominator_matrix = sij/down_matrix_1 + initial_theta*np.transpose(sij)/down_matrix_2
#            print(denominator_matrix)
            
            denominator = np.sum(denominator_matrix,axis=1)
            
            initial = si/denominator
            initial = initial/np.sum(initial)
#            print('gamma')
#            print(initial)
            # update theta
            # C_k
            up_matrix_C=np.zeros((num_player,num_player))
            for i in range(num_player):
                up_matrix_C[i]=initial
                
            up_matrix_C=up_matrix_C*sij
            
            down_matrix_C=np.zeros((num_player,num_player))
            
            for i in range(num_player):
                down_matrix_C[i]=initial[i]+initial_theta*initial
                
            C_k_matrix = up_matrix_C/down_matrix_C
#            print(C_k_matrix)
                
            C_k=np.sum(C_k_matrix)/(2*T) 
#            print(C_k)
            # theta
            initial_theta=(1/(2*C_k))+np.sqrt(1+1/(4*(C_k**2)))
#            print('theta')
#            print(initial_theta)
            
            # error
            if np.max(np.abs(last - initial))/np.max(np.abs(initial)) < error and np.abs(last_theta-initial_theta) < error:
#                print(np.max(np.abs(last - initial))/np.max(np.abs(gamma)))
#                print(np.abs(last_theta-initial_theta))
                print('Converge')
                estimation = initial
                print(estimation)
                estimation_theta=initial_theta
                print(estimation_theta)
                num_iter=k
                break
            
        return estimation, estimation_theta

    def Rao_Kupper_given_theta(self,iteration=1000,error=1e-9,theta=1.5):
        
        win_graph=self.win_graph
        tie_graph=self.tie_graph
        num_player=self.size
        
        gamma=np.ones(num_player)/num_player
        initial = gamma.copy()
        
        win=np.sum(win_graph,axis=1)
        tie=np.sum(tie_graph,axis=1)

        si=win+tie # s_i
        sij=win_graph+tie_graph
        T=np.sum(tie)/2
        
        for k in range(iteration):
            
            last = initial.copy()
            # update gamma
            # down matrix_1
            down_matrix_1=np.zeros((num_player,num_player))
            for i in range(num_player):
                down_matrix_1[i]=initial[i]+theta*initial
                
            down_matrix_2=np.zeros((num_player,num_player))
            for i in range(num_player):
                down_matrix_2[i]=theta*initial[i]+initial
                
            denominator_matrix = sij/down_matrix_1 + theta*np.transpose(sij)/down_matrix_2
#            print(denominator_matrix)
            
            denominator = np.sum(denominator_matrix,axis=1)
            
            initial = si/denominator
            initial = initial/np.sum(initial)

            # error
            if np.max(np.abs(last - initial))/np.max(np.abs(initial)) < error:
#                print(np.max(np.abs(last - initial))/np.max(np.abs(gamma)))
#                print(np.abs(last_theta-initial_theta))
                print('Converge')
                estimation = initial
                print(estimation)
                num_iter=k
                break
            
        return estimation

    def clm(self,iteration=50000,error=1e-6):
        
        initial_K =  np.ones(self.size)/self.size #gamma
        initial_hyperparameter = 1.5

        graph_4 = self.graph_4
        graph_3 = self.graph_3
        graph_2 = self.graph_2
        graph_1 = self.graph_1
        outcome_4_graph = np.sum(graph_4,axis=1)
        outcome_3_graph = np.sum(graph_3,axis=1)
        outcome_2_graph = np.sum(graph_2,axis=1)
        outcome_1_graph = np.sum(graph_1,axis=1)
        
        for K in range(iteration):
            last = initial_K.copy()
            current_probability = np.zeros((self.size,self.size))
            for i in range(self.size):
                current_probability[i] = initial_K[i]       # ij term = gamma_i
            
            transpose_current_probability = np.transpose(current_probability)   # ij term = gamma_j
                    
            down_1 = (graph_4 + graph_3)/((current_probability + initial_hyperparameter*transpose_current_probability))
            down_2 = ((graph_1 + graph_2)*initial_hyperparameter)/((transpose_current_probability + initial_hyperparameter*current_probability))
            down_3 = ((graph_2 + graph_3))/((transpose_current_probability + current_probability))

            down = down_1 + down_2 + down_3
            down_vector = np.sum(down,axis = 1)
            up_vector = outcome_4_graph + outcome_3_graph + outcome_2_graph

            initial_K = up_vector/down_vector
            initial_K = initial_K/(np.sum(initial_K))
            
            for i in range(self.size):
                current_probability[i] = initial_K[i]       # ij term = gamma_i
            
            transpose_current_probability = np.transpose(current_probability)   # ij term = gamma_j
            

            up_vector_hyper = np.sum(graph_3)
            down_vector_hyper_1 = (graph_3*1 + graph_4*1)*(transpose_current_probability)/(current_probability + initial_hyperparameter*transpose_current_probability)
            down_vector_hyper =  np.sum(down_vector_hyper_1)
            initial_hyperparameter = (up_vector_hyper/down_vector_hyper)+1
            
            if np.max(np.abs(last - initial_K)) < error:
                print('converge')
                estimation = initial_K
                estimation_theta=initial_hyperparameter
                
                break
            
        return estimation,estimation_theta
    
    def clm_given_theta(self,iteration=5000,error=1e-6,theta=2):
        
        initial_K =  np.ones(self.size)/self.size #gamma
        initial_hyperparameter = theta

        graph_4 = self.graph_4
        graph_3 = self.graph_3
        graph_2 = self.graph_2
        graph_1 = self.graph_1
        outcome_4_graph = np.sum(graph_4,axis=1)
        outcome_3_graph = np.sum(graph_3,axis=1)
        outcome_2_graph = np.sum(graph_2,axis=1)
        outcome_1_graph = np.sum(graph_1,axis=1)
        
        for K in range(iteration):
            last = initial_K.copy()
            current_probability = np.zeros((self.size,self.size))
            for i in range(self.size):
                current_probability[i] = initial_K[i]       # ij term = gamma_i
            
            transpose_current_probability = np.transpose(current_probability)   # ij term = gamma_j
                    
            down_1 = (graph_4 + graph_3)/((current_probability + initial_hyperparameter*transpose_current_probability))
            down_2 = ((graph_1 + graph_2)*initial_hyperparameter)/((transpose_current_probability + initial_hyperparameter*current_probability))
            down_3 = ((graph_2 + graph_3))/((transpose_current_probability + current_probability))

            down = down_1 + down_2 + down_3
            down_vector = np.sum(down,axis = 1)
            up_vector = outcome_4_graph + outcome_3_graph + outcome_2_graph

            initial_K = up_vector/down_vector
            initial_K = initial_K/(np.sum(initial_K))
            
            if np.max(np.abs(last - initial_K)) < error:
                print('converge')
                estimation = initial_K
                
                break
            
        return estimation

    def aclm(self,iteration=5000,error=1e-6):
        initial_K_2 = np.ones(self.size)/self.size
        initial_hyperparameter_2 = 20
        
        graph_4 = self.graph_4
        graph_3 = self.graph_3
        graph_2 = self.graph_2
        graph_1 = self.graph_1
        outcome_4_graph = np.sum(graph_4,axis=1)
        outcome_3_graph = np.sum(graph_3,axis=1)
        outcome_2_graph = np.sum(graph_2,axis=1)
        outcome_1_graph = np.sum(graph_1,axis=1)
        
        for K in range(iteration):
            last = initial_K_2.copy()
            current_ratio = np.zeros((self.size,self.size))
            current_probability = np.zeros((self.size,self.size))

            for i in range(self.size):
                current_ratio[i] = initial_K_2[i]/initial_K_2       # ij term = gamma_i/gamma_j
                current_probability[i] = initial_K_2[i]      # ij term = gamma_i

            transpose_current_probability = np.transpose(current_probability)   # ij term = gamma_j
            transpose_current_ratio = np.transpose(current_ratio)   # ij term = gamma_j/gamma_i

                    
            down_matrix_1 = (1 + ((initial_hyperparameter_2 * np.cbrt(transpose_current_ratio**2)/3)) + (2*(initial_hyperparameter_2 * np.cbrt(transpose_current_ratio)/3)))
            down_matrix_2 = transpose_current_probability + (initial_hyperparameter_2*np.cbrt((transpose_current_probability**2)*current_probability)) + (initial_hyperparameter_2*np.cbrt((current_probability**2)*transpose_current_probability)) + current_probability 
#            comparison_graph = binary_outcome_graph + np.transpose(binary_outcome_graph)

            down = (graph_4+graph_3+graph_2+graph_1) * down_matrix_1/down_matrix_2

            down_vector = np.sum(down,axis = 1)

            up_vector = outcome_4_graph + (2*outcome_3_graph/3) + (outcome_2_graph/3)

            initial_K_2 = up_vector/down_vector
            initial_K_2 = initial_K_2/(np.sum(initial_K_2))
            
            for i in range(self.size):
                current_ratio[i] = initial_K_2[i]/initial_K_2       # ij term = gamma_i/gamma_j
                current_probability[i] = initial_K_2[i]      # ij term = gamma_i

            transpose_current_probability = np.transpose(current_probability)   # ij term = gamma_j
            transpose_current_ratio = np.transpose(current_ratio)   # ij term = gamma_j/gamma_i


            up_vector_hyper = np.sum(graph_3)
            down_vector_hyper_matrix_2 =  (transpose_current_probability + (initial_hyperparameter_2*np.cbrt((transpose_current_probability**2)*current_probability)) + (initial_hyperparameter_2*np.cbrt((current_probability**2)*transpose_current_probability)) + current_probability) 
            down_vector_hyper_matrix_1 =  np.cbrt((transpose_current_probability**2)*current_probability) + np.cbrt((current_probability**2)*transpose_current_probability)
            down_vector_hyper = np.sum(((graph_3 + graph_4)) * down_vector_hyper_matrix_1/down_vector_hyper_matrix_2)

            initial_hyperparameter_2 = (up_vector_hyper/down_vector_hyper)
            
            if np.max(np.abs(last - initial_K_2))< error:
                estimation = initial_K_2
                print('converge')
                estimation_theta=initial_hyperparameter_2
                break
            
        return estimation,estimation_theta
    
    def aclm_given_theta(self,iteration=5000,error=1e-6,theta=2):
        initial_K_2 = np.ones(self.size)/self.size
        initial_hyperparameter_2 = theta
        
        graph_4 = self.graph_4
        graph_3 = self.graph_3
        graph_2 = self.graph_2
        graph_1 = self.graph_1
        outcome_4_graph = np.sum(graph_4,axis=1)
        outcome_3_graph = np.sum(graph_3,axis=1)
        outcome_2_graph = np.sum(graph_2,axis=1)
        outcome_1_graph = np.sum(graph_1,axis=1)
        
        for K in range(iteration):
            last = initial_K_2.copy()
            current_ratio = np.zeros((self.size,self.size))
            current_probability = np.zeros((self.size,self.size))

            for i in range(self.size):
                current_ratio[i] = initial_K_2[i]/initial_K_2       # ij term = gamma_i/gamma_j
                current_probability[i] = initial_K_2[i]      # ij term = gamma_i

            transpose_current_probability = np.transpose(current_probability)   # ij term = gamma_j
            transpose_current_ratio = np.transpose(current_ratio)   # ij term = gamma_j/gamma_i

                    
            down_matrix_1 = (1 + ((initial_hyperparameter_2 * np.cbrt(transpose_current_ratio**2)/3)) + (2*(initial_hyperparameter_2 * np.cbrt(transpose_current_ratio)/3)))
            down_matrix_2 = transpose_current_probability + (initial_hyperparameter_2*np.cbrt((transpose_current_probability**2)*current_probability)) + (initial_hyperparameter_2*np.cbrt((current_probability**2)*transpose_current_probability)) + current_probability 
#            comparison_graph = binary_outcome_graph + np.transpose(binary_outcome_graph)

            down = (graph_4+graph_3+graph_2+graph_1) * down_matrix_1/down_matrix_2

            down_vector = np.sum(down,axis = 1)

            up_vector = outcome_4_graph + (2*outcome_3_graph/3) + (outcome_2_graph/3)

            initial_K_2 = up_vector/down_vector
            initial_K_2 = initial_K_2/(np.sum(initial_K_2))
            
            
            if np.max(np.abs(last - initial_K_2))< error:
                estimation = initial_K_2
                print('converge')
                
                break
            
        return estimation
    
    def clm_5(self,iteration=50000,error=1e-6):
        
        initial_K =  np.ones(self.size)/self.size #gamma
        initial_theta_2 = 2
        initial_theta_1 = 2

        graph_5 = self.graph_5
        graph_4 = self.graph_4
        graph_3 = self.graph_3
        graph_2 = self.graph_2
        graph_1 = self.graph_1
        outcome_5_graph = np.sum(graph_5,axis=1)
        outcome_4_graph = np.sum(graph_4,axis=1)
        outcome_3_graph = np.sum(graph_3,axis=1)
        outcome_2_graph = np.sum(graph_2,axis=1)
        outcome_1_graph = np.sum(graph_1,axis=1)
        
        for K in range(iteration):
            last = initial_K.copy()
            current_probability = np.zeros((self.size,self.size))
            for i in range(self.size):
                current_probability[i] = initial_K[i]       # ij term = gamma_i
            
            transpose_current_probability = np.transpose(current_probability)   # ij term = gamma_j
                    
            down_1 = (graph_5 + graph_4)/((current_probability + initial_theta_2*transpose_current_probability))
            down_2 = ((graph_1 + graph_2)*initial_theta_2)/((transpose_current_probability + initial_theta_2*current_probability))
            down_3 = (graph_4 + graph_3)/(initial_theta_1*transpose_current_probability + current_probability)
            down_4 = (graph_2+graph_3)*initial_theta_1/(transpose_current_probability+initial_theta_1*current_probability)

            down = down_1 + down_2 + down_3 + down_4
            down_vector = np.sum(down,axis = 1)
            up_vector = outcome_5_graph + outcome_4_graph + outcome_3_graph + outcome_2_graph

            initial_K = up_vector/down_vector
            initial_K = initial_K/(np.sum(initial_K))
            #print(initial_K)
            
            # update theta
            
            for i in range(self.size):
                current_probability[i] = initial_K[i]       # ij term = gamma_i
            
            transpose_current_probability = np.transpose(current_probability)   # ij term = gamma_j
            
            # theta 2
            
            constant_left=np.sum(graph_2+graph_4)
            down_mtx=2*(graph_1+graph_2)*current_probability/(transpose_current_probability+initial_theta_2*current_probability)
            initial_theta_2=constant_left/np.sum(down_mtx)+initial_theta_1
            print('theta2')
            print(initial_theta_2)

            # theta 1
            
            C_k_mtx=transpose_current_probability*2*(graph_4+graph_3)/(current_probability+initial_theta_1*transpose_current_probability)
            C_k=np.sum(C_k_mtx)
            p1 = -np.sum(graph_2+graph_4)-2*np.sum(graph_3)-C_k*initial_theta_2
            p2=2*np.sum(graph_3)*initial_theta_2-C_k
            p3 = np.sum(graph_2+graph_4)+C_k*initial_theta_2
            p=[C_k,p1,p2,p3]
            roots=np.roots(p)
            print('roots')
            print(roots)
            real_roots=roots.real[abs(roots.imag)<1e-4]
            real_roots_positive = real_roots[0<real_roots]
            print('real roots')
            print(real_roots)
            real_roots_truncated = real_roots_positive[real_roots_positive<=initial_theta_2]
            
            # initial_theta_1=max(real_roots_truncated)
           
            
           
            if len(real_roots_truncated)!=0:
                initial_theta_1=max(real_roots_truncated)
            else:
                initial_theta_1=min(real_roots_positive)
                
            print('init theta 1')
            print(initial_theta_1)
            
            
            if np.max(np.abs(last - initial_K)) < error:
                print('converge')
                estimation = initial_K
            
                estimation_theta_1=initial_theta_1
                estimation_theta_2=initial_theta_2
                
                break
            
        return estimation,estimation_theta_1,estimation_theta_2
    
    def clm_5_given_theta(self,iteration=50000,error=1e-6,theta_1=3,theta_2=3):
        
        initial_K =  np.ones(self.size)/self.size #gamma
        initial_theta_2 = theta_2
        initial_theta_1 = theta_1

        graph_5 = self.graph_5
        graph_4 = self.graph_4
        graph_3 = self.graph_3
        graph_2 = self.graph_2
        graph_1 = self.graph_1
        outcome_5_graph = np.sum(graph_5,axis=1)
        outcome_4_graph = np.sum(graph_4,axis=1)
        outcome_3_graph = np.sum(graph_3,axis=1)
        outcome_2_graph = np.sum(graph_2,axis=1)
        outcome_1_graph = np.sum(graph_1,axis=1)
        
        for K in range(iteration):
            last = initial_K.copy()
            current_probability = np.zeros((self.size,self.size))
            for i in range(self.size):
                current_probability[i] = initial_K[i]       # ij term = gamma_i
            
            transpose_current_probability = np.transpose(current_probability)   # ij term = gamma_j
                    
            down_1 = (graph_5 + graph_4)/((current_probability + initial_theta_2*transpose_current_probability))
            down_2 = ((graph_1 + graph_2)*initial_theta_2)/((transpose_current_probability + initial_theta_2*current_probability))
            down_3 = (graph_4 + graph_3)/(initial_theta_1*transpose_current_probability + current_probability)
            down_4 = (graph_2+graph_3)*initial_theta_1/(transpose_current_probability+initial_theta_1*current_probability)

            down = down_1 + down_2 + down_3 + down_4
            down_vector = np.sum(down,axis = 1)
            up_vector = outcome_5_graph + outcome_4_graph + outcome_3_graph + outcome_2_graph

            initial_K = up_vector/down_vector
            initial_K = initial_K/(np.sum(initial_K))
            #print(initial_K)
            
            
            
            if np.max(np.abs(last - initial_K)) < error:
                print('converge')
                estimation = initial_K
                #estimation_theta_2=initial_theta_2
                
                break
            
        return estimation
    
    def aclm_5(self,iteration=50000,error=1e-6):
        
        initial_K =  np.ones(self.size)/self.size #gamma
        initial_theta_2 = 2
        initial_theta_1 = 2

        graph_5 = self.graph_5
        graph_4 = self.graph_4
        graph_3 = self.graph_3
        graph_2 = self.graph_2
        graph_1 = self.graph_1
        outcome_5_graph = np.sum(graph_5,axis=1)
        outcome_4_graph = np.sum(graph_4,axis=1)
        outcome_3_graph = np.sum(graph_3,axis=1)
        outcome_2_graph = np.sum(graph_2,axis=1)
        outcome_1_graph = np.sum(graph_1,axis=1)
        
        for K in range(iteration):
            last = initial_K.copy()
            current_probability = np.zeros((self.size,self.size))
            for i in range(self.size):
                current_probability[i] = initial_K[i]       # ij term = gamma_i
            
            transpose_current_probability = np.transpose(current_probability)   # ij term = gamma_j
            
            probability_graph = np.zeros((self.size,self.size))
            
            for i in range(self.size):
                probability_graph[i] = initial_K[i]/initial_K
                    
            down_w = graph_1+graph_2+graph_3+graph_4+graph_5
            down_para = 1+3*initial_theta_2*(1/probability_graph)**(1/4)/4+initial_theta_1*initial_theta_2*(1/probability_graph)**(1/2)/2+initial_theta_2*(1/probability_graph)**(3/4)/4
            P = current_probability+initial_theta_2*current_probability**(3/4)*transpose_current_probability**(1/4)+initial_theta_1*initial_theta_2*current_probability**(1/2)*transpose_current_probability**(1/2)+initial_theta_2*current_probability**(1/4)*transpose_current_probability**(3/4)+transpose_current_probability
            down_mtx = 2*down_w*down_para/P
            down_vector=np.sum(down_mtx,axis=1)
            
            up_vector = (4*outcome_5_graph + 3*outcome_4_graph + 2*outcome_3_graph + outcome_2_graph)/2

            initial_K = up_vector/down_vector
            initial_K = initial_K/(np.sum(initial_K))
            #print(initial_K)
            
            # update theta
            
            for i in range(self.size):
                current_probability[i] = initial_K[i]       
            
            transpose_current_probability = np.transpose(current_probability)
            
            # theta 1
            P_1 = current_probability+initial_theta_2*current_probability**(3/4)*transpose_current_probability**(1/4)+initial_theta_1*initial_theta_2*current_probability**(1/2)*transpose_current_probability**(1/2)+initial_theta_2*current_probability**(1/4)*transpose_current_probability**(3/4)+transpose_current_probability
            down_1_m = (2*graph_1+2*graph_2+graph_3)*initial_theta_2*current_probability**(1/2)*transpose_current_probability**(1/2)/P_1
            down_1=np.sum(down_1_m)
            initial_theta_1=np.sum(graph_3)/down_1
            print('theta')
            print(initial_theta_1)
            
            # theta 2
            P_2 = current_probability+initial_theta_2*current_probability**(3/4)*transpose_current_probability**(1/4)+initial_theta_1*initial_theta_2*current_probability**(1/2)*transpose_current_probability**(1/2)+initial_theta_2*current_probability**(1/4)*transpose_current_probability**(3/4)+transpose_current_probability
            down_2_m=(2*graph_1+2*graph_2+graph_3)*(current_probability**(3/4)*transpose_current_probability**(1/4)+current_probability**(1/4)*transpose_current_probability**(3/4)+initial_theta_1*current_probability**(1/2)*transpose_current_probability**(1/2))/P_2
            down_2=np.sum(down_2_m)
            initial_theta_2=np.sum(2*graph_2+graph_3)/down_2
            
            print(initial_theta_2)
            
            if np.max(np.abs(last - initial_K)) < error:
                print('converge')
                estimation = initial_K
                estimation_theta_1=initial_theta_1
                estimation_theta_2=initial_theta_2
                
                break
            
        return estimation,estimation_theta_1,estimation_theta_2

    
    def aclm_5_given_theta(self,iteration=50000,error=1e-6,theta_1=3,theta_2=3):
        
        initial_K =  np.ones(self.size)/self.size #gamma
        initial_theta_2 = theta_2
        initial_theta_1 = theta_1

        graph_5 = self.graph_5
        graph_4 = self.graph_4
        graph_3 = self.graph_3
        graph_2 = self.graph_2
        graph_1 = self.graph_1
        outcome_5_graph = np.sum(graph_5,axis=1)
        outcome_4_graph = np.sum(graph_4,axis=1)
        outcome_3_graph = np.sum(graph_3,axis=1)
        outcome_2_graph = np.sum(graph_2,axis=1)
        outcome_1_graph = np.sum(graph_1,axis=1)
        
        for K in range(iteration):
            last = initial_K.copy()
            current_probability = np.zeros((self.size,self.size))
            for i in range(self.size):
                current_probability[i] = initial_K[i]       # ij term = gamma_i
            
            transpose_current_probability = np.transpose(current_probability)   # ij term = gamma_j
            
            probability_graph = np.zeros((self.size,self.size))
            
            for i in range(self.size):
                probability_graph[i] = initial_K[i]/initial_K
                    
            down_w = graph_1+graph_2+graph_3+graph_4+graph_5
            down_para = 1+3*initial_theta_2*(1/probability_graph)**(1/4)/4+initial_theta_1*initial_theta_2*(1/probability_graph)**(1/2)/2+initial_theta_2*(1/probability_graph)**(3/4)/4
            P = current_probability+initial_theta_2*current_probability**(3/4)*transpose_current_probability**(1/4)+initial_theta_1*initial_theta_2*current_probability**(1/2)*transpose_current_probability**(1/2)+initial_theta_2*current_probability**(1/4)*transpose_current_probability**(3/4)+transpose_current_probability
            down_mtx = 2*down_w*down_para/P
            down_vector=np.sum(down_mtx,axis=1)
            
            up_vector = (4*outcome_5_graph + 3*outcome_4_graph + 2*outcome_3_graph + outcome_2_graph)/2

            initial_K = up_vector/down_vector
            initial_K = initial_K/(np.sum(initial_K))
            
            
            
            
            if np.max(np.abs(last - initial_K)) < error:
                print('converge')
                estimation = initial_K
                
                
                break
            
        return estimation

 
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        