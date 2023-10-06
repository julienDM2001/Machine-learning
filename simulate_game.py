from transition import compute_transition_matrix
import numpy as np
import matplotlib.pyplot as plt
from q_learn import q_learning
from transition import generate_arrays
from main import markovDecision
def simulate_game(strategy, layout, circle, n_iterations=10000):
    # Compute transition matrices for each dice

    P_safe, P_normal, P_risky = compute_transition_matrix(layout, circle)
    transition_matrices = [P_safe, P_normal, P_risky]
    number_turns = []
    total_turns = 0
    for _ in range(n_iterations):
        total_turns = 0
        state = 0  # initial state
        while state < len(layout)-1:  # until goal state is reached
            action = strategy[state] # get action according to strategy
            transition_matrix = transition_matrices[int(action - 1)]
            state = np.random.choice(len(layout), p=transition_matrix[state])
            if layout[state] == 3 and action == 2:
                total_turns += 1 if np.random.uniform(0, 1) < 0.5 else 2
            elif layout[state] == 3 and action == 3:
                total_turns += 2
            else:
                total_turns += 1
        number_turns.append(total_turns)
        # calculte the average number of turns
    return np.mean(number_turns)

# for a number of different layout of 15 squares, with 4 different trap each time compare the average number of turns for each strategy
# and plot the results

def plot_results(layouts, circle, n_iterations=100):
    resultsm = []
    resultss = []
    resultsn = []
    resultsr = []
    resultsrand = []
    resultsq = []
    for layout in layouts:
        # Compute optimal policy and Q-table
        expec, policy = markovDecision(layout, circle)
        # Simulate game
        resultm = simulate_game(policy, layout, circle, n_iterations)
        resultsm.append(resultm)
        expec1,policy1 = q_learning(layout, circle)
        resultq = simulate_game(policy1, layout, circle, n_iterations)
        resultsq.append(resultq)
        results = simulate_game([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], layout, circle, n_iterations)
        resultss.append(results)
        resultn = simulate_game([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2], layout, circle, n_iterations)
        resultsn.append(resultn)
        resultr = simulate_game([3,3,3,3,3,3,3,3,3,3,3,3,3,3,3], layout, circle, n_iterations)
        resultsr.append(resultr)
        resultrand = simulate_game([np.random.randint(1,4) for _ in range(15)], layout, circle, n_iterations)
        resultsrand.append(resultrand)

    # now plot the results to compare the x axis is the nomber of the layout
    # set the font size bigger
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(layouts)), resultsm, label='Markov')
    plt.plot(range(len(layouts)), resultss, label='Safe')
    plt.plot(range(len(layouts)), resultsn, label='Normal')
    plt.plot(range(len(layouts)), resultsr, label='Risky')
    plt.plot(range(len(layouts)), resultsrand, label='Random')
    plt.plot(range(len(layouts)), resultsq, label='Q-learning')
    # show every integer value on the x axis
    plt.xticks(range(len(layouts)), range(len(layouts)))
    plt.xlabel('layout number', fontsize=13)
    plt.ylabel('Average number of turns', fontsize=13)
    # increase the size of the legend
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    plt.show()


def simulate_state(strategy, layout, circle, n_iterations=10000):
    # Compute transition matrices for each dice

    P_safe, P_normal, P_risky = compute_transition_matrix(layout, circle)
    transition_matrices = [P_safe, P_normal, P_risky]
    number_turns = []
    number_mean = []
    for _ in range(n_iterations):
        number_turns = []
        for state in range(len(layout)-1):

            total_turns = 0
            while state < len(layout)-1:
                action = strategy[state]
                transition_matrix = transition_matrices[int(action - 1)]
                state = np.random.choice(len(layout), p=transition_matrix[state])
                if layout[state] == 3 and action == 2:
                    total_turns += 1 if np.random.uniform(0, 1) < 0.5 else 2
                elif layout[state] == 3 and action == 3:
                    total_turns += 2
                else:
                    total_turns += 1
            number_turns.append(total_turns)

        number_mean.append(number_turns)
    # calculte the average number of turns for each state
    print(np.mean(number_mean, axis=0))

    return np.mean(number_mean, axis=0)
