#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSMA Multi-arm bandit opponents for teh slot machine game.

In this python notebook you will learn about 

* The exploration-exploitation dilemma in reinforcement learning
* How multi-arm bandits 'home-in' on the best solution over time.

Notes:
    
    This python file has been setup to run in Spyter using its 'cell' approach
    to execution.  Click your cursor anywhere in a cell (indicated by the 
    dividing lines) and press Shift-Return.

"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



#import multi-arm bandit agents
from .agents import (EpsilonGreedy, 
                    AnnealingEpsilonGreedy,
                    ThompsonSamplingBeta,
                    OptimisticInitialValues,
                    UpperConfidenceBound)

#
from .environments import (custom_bandit_problem,
                           BernoulliCasino)



# Let's look at how we would organise our code to that we can run experiments

# First some simple utility functions to help us print out results...

def print_reward(agent):
    '''
    Utility function to print formatted results
    
    Parameters
    ----------
    agent : object
        Multi arm bandit agent.

    Returns
    -------
    None.

    '''
    print(f'Total reward: {agent.total_reward}')
    print('\nFinal Model:\n------')
    for bandit_index in range(len(agent._means)):
        print(f'Bandit {bandit_index + 1}:\t{agent._means[bandit_index]:.2f}')


def visualise_agent_actions(agent):
    '''
    Visualise the actions taken in a bar chart

    Params:
    -----
    agent : object
        Multi arm bandit agent.
    
    '''
    actions = agent.actions
    x = [i + 1 for i in range(actions.shape[0])]
    plt.bar(x, actions)
    plt.title('Histogram of Actions Taken by Algorithm')
    plt.xlabel('Arm')
    plt.ylabel('Number of times each arm was selected')
    plt.show()


def the_agent(epsilon=0.2, budget=30, random_seed=42):
    '''
    The agent formerly known as epsilon-greedy
    
    Params:
    -------
    epsilon: float
       exploration parameter.  percentage of bandit rounds that are random
    
    budget: int
        Total number of rounds the agent will play
        
    random_seed: int
        Parameter to control random sampling to ensure you get a repeated 
        result
        
    Returns:
    -------
        None.
    
    '''
    print('------\nAgent: Formerly known as Epsilon-Greedy')
    
    #to reproduce the result set a random seed
    np.random.seed(seed=random_seed)

    #create environment
    bandit_arms = custom_bandit_problem(0.2, 0.5, 0.3, 0.75, 0.3)
    environment = BernoulliCasino(bandits=bandit_arms)

    #create agent and solve
    agent = EpsilonGreedy(epsilon=epsilon, budget=budget, environment=environment)
    agent.solve()
    
    #print out formatted results
    print_reward(agent)
    visualise_agent_actions(agent)
    


def the_reverend(budget=30, random_seed=42):
    '''
    The reverend Thompson
    
    Params:
    -------
    
    budget: int
        Total number of rounds the agent will play
        
    random_seed: int
        Parameter to control random sampling to ensure you get a repeated 
        result
        
    Returns:
    -------
        None.
    
    '''
    print('------\nAgent: The Reverend Thompson')
    
    #to reproduce the result set a random seed
    np.random.seed(seed=random_seed)

    #create environment
    bandit_arms = custom_bandit_problem(0.2, 0.5, 0.3, 0.75, 0.3)
    environment = BernoulliCasino(bandits=bandit_arms)

    #create agent and solve
    agent = ThompsonSamplingBeta(budget=budget, environment=environment)
    agent.solve()
    
    #print out formatted results
    print_reward(agent)
    visualise_agent_actions(agent)


def the_master(budget=30, random_seed=42):
    '''
    The Uncertainty Master
    
    Params:
    -------    
    budget: int
        Total number of rounds the agent will play
        
    random_seed: int
        Parameter to control random sampling to ensure you get a repeated 
        result
        
    Returns:
    -------
        None.
    
    '''
    print('------\nAgent: The Uncertainty Master')
    
    #to reproduce the result set a random seed
    np.random.seed(seed=random_seed)

    #create environment
    bandit_arms = custom_bandit_problem(0.2, 0.5, 0.3, 0.75, 0.3)
    environment = BernoulliCasino(bandits=bandit_arms)

    #create agent and solve
    agent = UpperConfidenceBound(budget=budget, environment=environment)
    agent.solve()
    
    #print out formatted results
    print_reward(agent)
    visualise_agent_actions(agent)
    
    
def the_optimist(budget=30, random_seed=42):
    '''
    The Uncertainty Master
    
    Params:
    -------
    epsilon: float
       exploration parameter.  percentage of bandit rounds that are random
    
    budget: int
        Total number of rounds the agent will play
        
    random_seed: int
        Parameter to control random sampling to ensure you get a repeated 
        result
        
    Returns:
    -------
        None.
    
    '''
    print('------\nAgent: The Uncertainty Master')
    
    #to reproduce the result set a random seed
    np.random.seed(seed=random_seed)

    #create environment
    bandit_arms = custom_bandit_problem(0.2, 0.5, 0.3, 0.75, 0.3)
    environment = BernoulliCasino(bandits=bandit_arms)

    #create agent and solve
    agent = OptimisticInitialValues(budget=budget, environment=environment)
    agent.solve()
    
    #print out formatted results
    print_reward(agent)
    visualise_agent_actions(agent)
