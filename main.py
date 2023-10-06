import numpy as np
from transition import compute_transition_matrix
# test the function above by generating transition matrix for vectors of size 15 composed of random value in [0,1,2,3,4] the first and last value are always 0



def markovDecision(layout, circle):


    n_states = 15


    # Get transition probabilities
    P_safe, P_normal, P_risky = compute_transition_matrix(layout, circle)

    # Value iteration
    V = np.zeros(n_states)
    prisons = [i for i, x in enumerate(layout) if x == 3]
    Dice = np.zeros(n_states - 1)
    n_ite = 0
    while True:
        V_new = np.zeros(n_states)
        n_ite += 1

        for s in range(n_states - 1):
            q_safe = np.sum(P_safe[s] * V)
            q_normal = np.sum(P_normal[s] * V) + 0.5 * np.sum(P_normal[s][prisons])
            q_risky = np.sum(P_risky[s] * V) +  np.sum(P_normal[s][prisons])
            V_new[s] = 1 + min(q_safe, q_normal, q_risky)

            if V_new[s] == 1 + q_safe:
                Dice[s] = 1
            elif V_new[s] == 1 + q_normal:
                Dice[s] = 2
            else:
                Dice[s] = 3

        if np.allclose(V_new, V):
            V = V_new
            break

        V = V_new

    Expec = V[:-1]
    return [Expec, Dice]
layout = [0,0,0,0,4,0,0,0,2,0,0,1,3,0,0]
print(markovDecision(layout, False))
print(markovDecision(layout, True))
