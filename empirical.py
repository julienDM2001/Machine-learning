import numpy as np
import random
import matplotlib.pyplot as plt
from main import markovDecision
from simulate_game import simulate_state
from transition import generate_arrays
from q_learn import q_learning
def plot_results(layouts, circle, n_iterations=100):
    resultsm = []
    resultxq = []
        # Compute optimal policy and Q-table
    expec, policy = markovDecision(layouts, circle)
    expec2, policy2 = q_learning(layouts, circle)
        # Simulate game
    resultm = simulate_state(policy, layouts, circle, n_iterations)
    resultq = simulate_state(policy2, layouts, circle, n_iterations)
    resultsm.append(resultm)
    resultxq.append(resultq)



    # now plot the results to compare expec and resultm
    plt.figure(figsize=(12, 10))
    plt.plot(resultm, label='Markov Decision')
    plt.plot(resultq, label='Q Learning')
    plt.xticks(range(len(resultm)), range(len(resultm)))
    plt.yticks(range(0,15,1))
    plt.xlabel('layout number', fontsize=14)
    plt.ylabel('Average number of turns', fontsize=14)
    # increase the size of the legend
    plt.legend( bbox_to_anchor=(1, 1), ncol=1)
    plt.show()

layouts = [0,1,0,0,0,0,3,4,2,0,0,0,0,0,0]
plot_results(layouts, False, 1000000)
plot_results(layouts, True, 1000000)
