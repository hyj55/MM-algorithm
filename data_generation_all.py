#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 14:28:03 2022

@author: jiangyunhui
"""

import numpy as np

class data_generation:
    
    def __init__(self, size, num_game, dynamic_range, sparsity, **theta):
        
        """
        Parameters
        ----------
        size : The  number of players involved in the comparison.
        
        theta : Global parameter ("threshold" parameter), input should be larger than 1 and follow the format:
            when J=5:
                theta_1 = theta_1, theta_2 = theta_2
            when J=4,3:
                theta = theta
            when J<3:
                theta=0

        Returns
        -------
        generate_gamma: gamma.
        data_generations: graphs, *players, gamma

        """
        self.dynamic_range = dynamic_range
        self.sparsity = sparsity
        self.size = size
        
        for key, value in theta.items():
            self.__dict__[key] = value
        
        
        self.num_game = num_game
        self.matrix_1 = np.ones((self.size,self.size))
        self.matrix_2 = np.zeros((self.size,self.size))
    
        for i in range(self.size):
            for j in range(self.size):
                if i > j:
                    self.matrix_1[i,j], self.matrix_2[i,j] = 0,1
                elif i == j:
                    self.matrix_1[i,j], self.matrix_2[i,j] = 0,0
                    
                    
    def generate_gamma(self):
        
        def reparameterize(parameter):
            return np.exp(parameter)/np.sum(np.exp(parameter))
        
        parameter_left_one = np.random.uniform(-self.dynamic_range, self.dynamic_range, size=(self.size-1))
        parameter = np.insert(parameter_left_one,0,0)
        gamma = reparameterize(parameter)
        return gamma
    
    def data_generation_rao(self):
        
        num_subject=self.size # num of players
        gamma=self.generate_gamma()
        theta=self.theta
        
        Players=np.array(range(num_subject))
        
        matrix_1 = self.matrix_1
        matrix_2 = self.matrix_2
                    
        probability=self.sparsity
        comparison_graph = np.zeros((num_subject,num_subject))
        for i in range(num_subject):
            pro = np.random.uniform(low = 0, high = 1-probability, size = (num_subject))
            comparison_graph[i] = np.random.binomial(1, pro + probability, size = (num_subject)) # prob of having a game between i and j
        
        comparison_graph = comparison_graph*matrix_1
        comparison_graph = comparison_graph + np.transpose(comparison_graph)  # 1 for having a game between i,j; 0 for not.

        win_prob_graph = np.zeros((num_subject,num_subject)) 
        for i in range(num_subject):
            win_prob_graph[i] = gamma[i]/(gamma[i]+theta*gamma)
            
    #    print(win_prob_graph)
            
        tie_given_win_prob_graph_cond = 1-win_prob_graph
    #    print(tie_given_win_prob_graph_cond)
        
        loss_graph_under_no_win_graph=np.zeros((num_subject,num_subject))
        for i in range(num_subject):
            loss_graph_under_no_win_graph[i] = gamma/(theta*gamma[i]+gamma)
    #    print(loss_graph_under_no_win_graph)
            
        tie_vs_loss_graph_under_no_win_graph_prob = 1-(loss_graph_under_no_win_graph/tie_given_win_prob_graph_cond)
    #    print(tie_vs_loss_graph_under_no_win_graph)
            
        win_graph=np.zeros((num_subject,num_subject))
        tie_graph=np.zeros((num_subject,num_subject))
            
        for i in range(self.num_game):
            
            win_vs_notwin_graph = np.random.binomial(1, win_prob_graph, size = (num_subject,num_subject))  # win is one 
            notwin_vs_win_graph = 1 - win_vs_notwin_graph       # not win is one
            
            tie_vs_loss_graph_under_no_win_graph = np.random.binomial(1, tie_vs_loss_graph_under_no_win_graph_prob, size = (num_subject,num_subject))  # tie is one
            tie_vs_loss_graph_under_no_win_graph = tie_vs_loss_graph_under_no_win_graph + 1    # tie is two while loss is one
            
            outcome_graph = tie_vs_loss_graph_under_no_win_graph * notwin_vs_win_graph + win_vs_notwin_graph * 3  # loss is one, tie is two, win is three
            outcome_graph = outcome_graph*matrix_1
            
            outcome_graph_2 =  4 - np.transpose(outcome_graph)    # loss is 1, tie is 2, win is 3 (i beats j)
            outcome_graph_2 = outcome_graph_2*matrix_2
            
            total_outcome_graph = outcome_graph + outcome_graph_2
            
            graph = total_outcome_graph*comparison_graph
            
            win_graph = win_graph+1*(graph==3)
            tie_graph = tie_graph+1*(graph==2)
        
        
        return Players, win_graph, tie_graph, gamma
    
    def data_generation_davi(self):
        
        # data generation
        num_subject=self.size # num of players
        gamma=self.generate_gamma()
        theta=self.theta
        Players=np.array(range(num_subject))
        
        matrix_1 = self.matrix_1
        matrix_2 = self.matrix_2
    
        probability=self.sparsity    # prob of having a game between i and j
     
        comparison_graph = np.zeros((num_subject,num_subject))
        for i in range(num_subject):
            pro = np.random.uniform(low = 0, high = 1-probability, size = (num_subject))
            comparison_graph[i] = np.random.binomial(1, pro + probability, size = (num_subject)) # prob of having a game between i and j
        
        comparison_graph = comparison_graph*matrix_1
        comparison_graph = comparison_graph + np.transpose(comparison_graph)  # 1 for having a game between i,j; 0 for not.
        
        
        probability_graph = np.zeros((num_subject,num_subject)) 
        for i in range(num_subject):
            probability_graph[i] = gamma[i]/gamma  # ij term = gamma_i/gamma_j
            
        win_graph=np.zeros((num_subject,num_subject))
        tie_graph=np.zeros((num_subject,num_subject))
            
        for i in range(self.num_game):
            
            win_vs_notwin_graph = probability_graph/(probability_graph + 1 + theta*np.sqrt(probability_graph))   # prob of i wins j for each ij
            win_vs_notwin_graph = np.random.binomial(1, win_vs_notwin_graph, size = (num_subject,num_subject))  # win is one 
            notwin_vs_win_graph = 1 - win_vs_notwin_graph       # not win is one
            
            tie_vs_loss_graph_under_no_win_graph = theta*np.sqrt(probability_graph)/(1 + theta*np.sqrt(probability_graph))   
            tie_vs_loss_graph_under_no_win_graph = np.random.binomial(1, tie_vs_loss_graph_under_no_win_graph, size = (num_subject,num_subject))  # tie is one
            tie_vs_loss_graph_under_no_win_graph = tie_vs_loss_graph_under_no_win_graph + 1    # tie is two while loss is one
            
            outcome_graph = tie_vs_loss_graph_under_no_win_graph * notwin_vs_win_graph + win_vs_notwin_graph * 3  # loss is one, tie is two, win is three
            outcome_graph = outcome_graph*matrix_1
            
            outcome_graph_2 =  4 - np.transpose(outcome_graph)    # loss is 1, tie is 2, win is 3 (i beats j)
            outcome_graph_2 = outcome_graph_2*matrix_2
            
            total_outcome_graph = outcome_graph + outcome_graph_2
            graph = total_outcome_graph*comparison_graph
            
            win_graph = win_graph+1*(graph==3)
            tie_graph = tie_graph+1*(graph==2)
        
        
        return Players, win_graph, tie_graph, gamma

    def generate_four_data_clm(self):
        
        graph_1=np.zeros((self.size,self.size))
        graph_2=np.zeros((self.size,self.size))
        graph_3=np.zeros((self.size,self.size))
        graph_4=np.zeros((self.size,self.size))
        num_game=self.num_game
        
        comparison_graph = np.random.binomial(1, self.sparsity, size = (self.size,self.size))
        comparison_graph = comparison_graph*self.matrix_1
        comparison_graph = comparison_graph + np.transpose(comparison_graph)
        self.comparison_graph = comparison_graph
    
    
        probability_graph = np.zeros((self.size,self.size))
        gamma = self.generate_gamma()
        for i in range(self.size):
            probability_graph[i] = gamma[i]/gamma
    
        outcomes_4 = 1/(1+self.theta*(1/probability_graph))
        outcomes_3 = (self.theta - 1)/((probability_graph + 1)*(1+self.theta*(1/probability_graph)))
        outcomes_2 = (self.theta - 1)/((probability_graph + 1)*(self.theta+(1/probability_graph)))
        outcomes_1 = 1/(1+self.theta*(probability_graph))
    
        for i in range(num_game):
            
            A4_vs_1_2_3_graph = np.random.binomial(1, outcomes_4, size = (self.size,self.size)) # 4 is presented as one
            A1_2_3_vs_4_graph = 1 - A4_vs_1_2_3_graph       # 1,2,3 are presented as one
                    
            A3_vs_1_2_under_no_4_graph_prob = outcomes_3/(1 - outcomes_4)
            A3_vs_1_2_under_no_4_graph = np.random.binomial(1, A3_vs_1_2_under_no_4_graph_prob, size = (self.size,self.size)) # 3 is presented as one
            A1_2_vs_3_under_no_4_graph = 1 - A3_vs_1_2_under_no_4_graph  # 1,2 are presented as one
        
            A2_vs_1_under_no_3_4_graph_prob = outcomes_2/(1 - outcomes_4 - outcomes_3)
            A2_vs_1_under_no_3_4_graph = np.random.binomial(1, A2_vs_1_under_no_3_4_graph_prob, size = (self.size,self.size)) 
            A1_vs_2_under_no_3_4_graph = 1 - A2_vs_1_under_no_3_4_graph
                    
            outcome_graph = (A4_vs_1_2_3_graph*4) + (A3_vs_1_2_under_no_4_graph*A1_2_3_vs_4_graph*3) + (A2_vs_1_under_no_3_4_graph*(A1_2_3_vs_4_graph * A1_2_vs_3_under_no_4_graph)*2) + (A1_2_3_vs_4_graph * A1_2_vs_3_under_no_4_graph*A1_vs_2_under_no_3_4_graph)
            outcome_graph = outcome_graph*self.matrix_1
                    
            outcome_graph_2 =  5 - np.transpose(outcome_graph)
            outcome_graph_2 = outcome_graph_2*self.matrix_2
                    
            total_outcome_graph = outcome_graph + outcome_graph_2
            graph = total_outcome_graph*comparison_graph
        
            graph_4 = graph_4+(graph == 4)
            graph_3 = graph_3+(graph == 3)
            graph_2 = graph_2+(graph == 2)
            graph_1 = graph_1+(graph == 1)
                

        return graph_4, graph_3, graph_2, graph_1, gamma
    
    
    def generate_four_data_aclm(self):
        
        graph_1=np.zeros((self.size,self.size))
        graph_2=np.zeros((self.size,self.size))
        graph_3=np.zeros((self.size,self.size))
        graph_4=np.zeros((self.size,self.size))
        num_game=self.num_game
        
        comparison_graph = np.random.binomial(1, self.sparsity, size = (self.size,self.size))
        comparison_graph = comparison_graph*self.matrix_1
        comparison_graph = comparison_graph + np.transpose(comparison_graph)
        self.comparison_graph = comparison_graph
    
    
        probability_graph = np.zeros((self.size,self.size))
        gamma = self.generate_gamma()
        for i in range(self.size):
            probability_graph[i] = gamma[i]/gamma
    
        outcomes_1 = 1/(1 + self.theta*np.cbrt(probability_graph) + self.theta*np.cbrt(probability_graph**2) + probability_graph)
        outcomes_2 = (self.theta*np.cbrt(probability_graph))/(1 + self.theta*np.cbrt(probability_graph) + self.theta*np.cbrt(probability_graph**2) + probability_graph)
        outcomes_3 = (self.theta*np.cbrt(probability_graph**2))/(1 + self.theta*np.cbrt(probability_graph) + self.theta*np.cbrt(probability_graph**2) + probability_graph)
        outcomes_4 = (probability_graph)/(1 + self.theta*np.cbrt(probability_graph) + self.theta*np.cbrt(probability_graph**2) + probability_graph)
    
        for i in range(num_game):
    
            A4_vs_1_2_3_graph = np.random.binomial(1, outcomes_4, size = (self.size,self.size)) # 4 is presented as one
            A1_2_3_vs_4_graph = 1 - A4_vs_1_2_3_graph       # 1,2,3 are presented as one
                    
            A3_vs_1_2_under_no_4_graph_prob = outcomes_3/(1 - outcomes_4)
            A3_vs_1_2_under_no_4_graph = np.random.binomial(1, A3_vs_1_2_under_no_4_graph_prob, size = (self.size,self.size)) # 3 is presented as one
            A1_2_vs_3_under_no_4_graph = 1 - A3_vs_1_2_under_no_4_graph  # 1,2 are presented as one
        
            A2_vs_1_under_no_3_4_graph_prob = outcomes_2/(1 - outcomes_4 - outcomes_3)
            A2_vs_1_under_no_3_4_graph = np.random.binomial(1, A2_vs_1_under_no_3_4_graph_prob, size = (self.size,self.size)) 
            A1_vs_2_under_no_3_4_graph = 1 - A2_vs_1_under_no_3_4_graph
                    
            outcome_graph = (A4_vs_1_2_3_graph*4) + (A3_vs_1_2_under_no_4_graph*A1_2_3_vs_4_graph*3) + (A2_vs_1_under_no_3_4_graph*(A1_2_3_vs_4_graph * A1_2_vs_3_under_no_4_graph)*2) + (A1_2_3_vs_4_graph * A1_2_vs_3_under_no_4_graph*A1_vs_2_under_no_3_4_graph)
            outcome_graph = outcome_graph*self.matrix_1
                    
            outcome_graph_2 =  5 - np.transpose(outcome_graph)
            outcome_graph_2 = outcome_graph_2*self.matrix_2
                    
            total_outcome_graph = outcome_graph + outcome_graph_2
            graph = total_outcome_graph*comparison_graph
        
            graph_4 = graph_4+(graph == 4)
            graph_3 = graph_3+(graph == 3)
            graph_2 = graph_2+(graph == 2)
            graph_1 = graph_1+(graph == 1)
                
        return graph_4, graph_3, graph_2, graph_1, gamma
    
    def generate_five_data_clm(self):
        
        graph_1=np.zeros((self.size,self.size))
        graph_2=np.zeros((self.size,self.size))
        graph_3=np.zeros((self.size,self.size))
        graph_4=np.zeros((self.size,self.size))
        graph_5=np.zeros((self.size,self.size))
        num_game=self.num_game
        
        comparison_graph = np.random.binomial(1, self.sparsity, size = (self.size,self.size))
        comparison_graph = comparison_graph*self.matrix_1
        comparison_graph = comparison_graph + np.transpose(comparison_graph)
        self.comparison_graph = comparison_graph
    
    
        probability_graph = np.zeros((self.size,self.size))
        gamma = self.generate_gamma()
        for i in range(self.size):
            probability_graph[i] = gamma[i]/gamma
    

        outcomes_1 = 1/(1+self.theta_2*(probability_graph))
        outcomes_2 = (self.theta_2 - self.theta_1)/((self.theta_1*probability_graph + 1)*(self.theta_2+(1/probability_graph)))
        outcomes_3 = (self.theta_1**2 - 1)/((self.theta_1 + probability_graph)*(self.theta_1+(1/probability_graph)))
        outcomes_4 = (self.theta_2-self.theta_1)/((1+self.theta_1*(1/probability_graph))*(probability_graph+self.theta_2))
        outcomes_5 = 1/(1+self.theta_2*(1/probability_graph))
    
        for i in range(num_game):
            
            A5_vs_1_2_3_4_graph=np.random.binomial(1, outcomes_5,size=(self.size,self.size))
            A1_2_3_4_vs_5_graph=1-A5_vs_1_2_3_4_graph
            
            A4_vs_1_2_3_under_no_5_graph_prob = outcomes_4/(1-outcomes_5) # 4 is presented as one
            A4_vs_1_2_3_under_no_5_graph = np.random.binomial(1, A4_vs_1_2_3_under_no_5_graph_prob, size=(self.size,self.size))
            A1_2_3_vs_4_under_no_5_graph = 1 - A4_vs_1_2_3_under_no_5_graph       # 1,2,3 are presented as one
                    
            A3_vs_1_2_under_no_4_5_graph_prob = outcomes_3/(1 - outcomes_5 - outcomes_4)
            A3_vs_1_2_under_no_4_5_graph = np.random.binomial(1, A3_vs_1_2_under_no_4_5_graph_prob, size = (self.size,self.size)) # 3 is presented as one
            A1_2_vs_3_under_no_4_5_graph = 1 - A3_vs_1_2_under_no_4_5_graph  # 1,2 are presented as one
        
            A2_vs_1_under_no_3_4_5_graph_prob = outcomes_2/(1 - outcomes_5 - outcomes_4 - outcomes_3)
            A2_vs_1_under_no_3_4_5_graph = np.random.binomial(1, A2_vs_1_under_no_3_4_5_graph_prob, size = (self.size,self.size)) 
            A1_vs_2_under_no_3_4_5_graph = 1 - A2_vs_1_under_no_3_4_5_graph
                    
            outcome_graph = A5_vs_1_2_3_4_graph*5 + (A4_vs_1_2_3_under_no_5_graph*A1_2_3_4_vs_5_graph*4) + (A3_vs_1_2_under_no_4_5_graph*A1_2_3_4_vs_5_graph*A1_2_3_vs_4_under_no_5_graph*3) + (A2_vs_1_under_no_3_4_5_graph*(A1_2_3_4_vs_5_graph*A1_2_3_vs_4_under_no_5_graph * A1_2_vs_3_under_no_4_5_graph)*2) + (A1_2_3_4_vs_5_graph*A1_2_3_vs_4_under_no_5_graph * A1_2_vs_3_under_no_4_5_graph*A1_vs_2_under_no_3_4_5_graph)
            outcome_graph = outcome_graph*self.matrix_1
                    
            outcome_graph_2 =  6 - np.transpose(outcome_graph)
            outcome_graph_2 = outcome_graph_2*self.matrix_2
                    
            total_outcome_graph = outcome_graph + outcome_graph_2
            graph = total_outcome_graph*comparison_graph
        
            graph_5 = graph_5+(graph == 5)
            graph_4 = graph_4+(graph == 4)
            graph_3 = graph_3+(graph == 3)
            graph_2 = graph_2+(graph == 2)
            graph_1 = graph_1+(graph == 1)
                

        return graph_5, graph_4, graph_3, graph_2, graph_1, gamma
    
    
    def generate_five_data_aclm(self):
        
        graph_1=np.zeros((self.size,self.size))
        graph_2=np.zeros((self.size,self.size))
        graph_3=np.zeros((self.size,self.size))
        graph_4=np.zeros((self.size,self.size))
        graph_5=np.zeros((self.size,self.size))
        num_game=self.num_game
        
        comparison_graph = np.random.binomial(1, self.sparsity, size = (self.size,self.size))
        comparison_graph = comparison_graph*self.matrix_1
        comparison_graph = comparison_graph + np.transpose(comparison_graph)
        self.comparison_graph = comparison_graph
    
    
        probability_graph = np.zeros((self.size,self.size))
        gamma = self.generate_gamma()
        for i in range(self.size):
            probability_graph[i] = gamma[i]/gamma
    

        outcomes_1 = 1/(probability_graph+self.theta_2*probability_graph**(3/4)+self.theta_1*self.theta_2*probability_graph**(1/2)+self.theta_2*probability_graph**(1/4)+1)
        outcomes_2 = self.theta_2*probability_graph**(1/4)/(probability_graph+self.theta_2*probability_graph**(3/4)+self.theta_1*self.theta_2*probability_graph**(1/2)+self.theta_2*probability_graph**(1/4)+1)
        outcomes_3 = self.theta_1*self.theta_2*probability_graph**(1/2)/(probability_graph+self.theta_2*probability_graph**(3/4)+self.theta_1*self.theta_2*probability_graph**(1/2)+self.theta_2*probability_graph**(1/4)+1)
        outcomes_4 = self.theta_2*probability_graph**(3/4)/(probability_graph+self.theta_2*probability_graph**(3/4)+self.theta_1*self.theta_2*probability_graph**(1/2)+self.theta_2*probability_graph**(1/4)+1)
        outcomes_5 = probability_graph/(probability_graph+self.theta_2*probability_graph**(3/4)+self.theta_1*self.theta_2*probability_graph**(1/2)+self.theta_2*probability_graph**(1/4)+1)
    
        for i in range(num_game):
            
            A5_vs_1_2_3_4_graph=np.random.binomial(1, outcomes_5,size=(self.size,self.size))
            A1_2_3_4_vs_5_graph=1-A5_vs_1_2_3_4_graph
            
            A4_vs_1_2_3_under_no_5_graph_prob = outcomes_4/(1-outcomes_5) # 4 is presented as one
            A4_vs_1_2_3_under_no_5_graph = np.random.binomial(1, A4_vs_1_2_3_under_no_5_graph_prob, size=(self.size,self.size))
            A1_2_3_vs_4_under_no_5_graph = 1 - A4_vs_1_2_3_under_no_5_graph       # 1,2,3 are presented as one
                    
            A3_vs_1_2_under_no_4_5_graph_prob = outcomes_3/(1 - outcomes_5 - outcomes_4)
            A3_vs_1_2_under_no_4_5_graph = np.random.binomial(1, A3_vs_1_2_under_no_4_5_graph_prob, size = (self.size,self.size)) # 3 is presented as one
            A1_2_vs_3_under_no_4_5_graph = 1 - A3_vs_1_2_under_no_4_5_graph  # 1,2 are presented as one
        
            A2_vs_1_under_no_3_4_5_graph_prob = outcomes_2/(1 - outcomes_5 - outcomes_4 - outcomes_3)
            A2_vs_1_under_no_3_4_5_graph = np.random.binomial(1, A2_vs_1_under_no_3_4_5_graph_prob, size = (self.size,self.size)) 
            A1_vs_2_under_no_3_4_5_graph = 1 - A2_vs_1_under_no_3_4_5_graph
                    
            outcome_graph = A5_vs_1_2_3_4_graph*5 + (A4_vs_1_2_3_under_no_5_graph*A1_2_3_4_vs_5_graph*4) + (A3_vs_1_2_under_no_4_5_graph*A1_2_3_4_vs_5_graph*A1_2_3_vs_4_under_no_5_graph*3) + (A2_vs_1_under_no_3_4_5_graph*(A1_2_3_4_vs_5_graph*A1_2_3_vs_4_under_no_5_graph * A1_2_vs_3_under_no_4_5_graph)*2) + (A1_2_3_4_vs_5_graph*A1_2_3_vs_4_under_no_5_graph * A1_2_vs_3_under_no_4_5_graph*A1_vs_2_under_no_3_4_5_graph)
            outcome_graph = outcome_graph*self.matrix_1
                    
            outcome_graph_2 =  6 - np.transpose(outcome_graph)
            outcome_graph_2 = outcome_graph_2*self.matrix_2
                    
            total_outcome_graph = outcome_graph + outcome_graph_2
            graph = total_outcome_graph*comparison_graph
        
            graph_5 = graph_5+(graph == 5)
            graph_4 = graph_4+(graph == 4)
            graph_3 = graph_3+(graph == 3)
            graph_2 = graph_2+(graph == 2)
            graph_1 = graph_1+(graph == 1)
                

        return graph_5, graph_4, graph_3, graph_2, graph_1, gamma
    
    



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
