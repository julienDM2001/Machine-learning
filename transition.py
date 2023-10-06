import numpy as np
import random
def compute_transition_matrix(layout,circle = False):
    P_safe = np.array([1 / 2, 1 / 2]) # transition probabilities for safe dice
    P_normal = np.array([1 / 3, 1 / 3, 1 / 3]) # transition probabilities for normal dice
    P_risky = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4]) # transition probabilities for risky dice

    matrix_safe = np.zeros((15, 15))
    matrix_normal = np.zeros((15, 15))
    matrix_risky = np.zeros((15, 15))

    for s in range(0,15): # for each state
        for step, prob in enumerate(P_safe): # calculate the transtition probability for the safe dice
            if s == 9 and step == 1: # if the state is 9 and the step is 1 then the next state is 14
                next_s = 14
                matrix_safe[s][next_s] += prob
            elif s == 2 and step > 0: # handle the fast lane
                prob /= 2
                next_s = 10
                matrix_safe[s,next_s] += prob
                next_s = 3
                matrix_safe[s,next_s] += prob
            else: # normal scenario
                next_s = s + step
                next_s = min(14, next_s)
                matrix_safe[s,next_s] += prob



        for step, prob in enumerate(P_normal): # calculate the transtition probability for the normal dice
            if s == 8 and step == 2: # if the state is 8 and the step is 2 then the next state is 14
                next_s = 14
                matrix_normal[s][next_s] += prob
                continue
            elif s == 9 and step in [1,2]: # if the state is 9 and the step is 1 or 2 then the next state is 14 + step - 1
                if not circle or step == 1:
                    next_s = 14
                    matrix_normal[s][next_s] += prob
                    continue
                if circle and step == 2:
                    next_s = 0
                    matrix_normal[s][next_s] += prob
                    continue
            elif s == 2 and step > 0: # handle the fast lane
                prob /= 2
                next_s = 10 + (step - 1)
                if layout[next_s] in [0,3]: # normal or prison square
                    matrix_normal[s][next_s] += prob
                elif layout[next_s] == 1: # handle type 1 trap
                    matrix_normal[s][next_s] += prob/2
                    next_s = 0
                    matrix_normal[s][next_s] += prob/2
                elif layout[next_s] == 2: # handle type 2 trap
                    matrix_normal[s][next_s] += prob / 2
                    if next_s == 10: # special scenario for the fast lane
                        next_s = 0
                        matrix_normal[s][next_s] += prob/2
                    elif next_s == 11: # special scenario for the fast lane
                        next_s = 1
                        matrix_normal[s][next_s] += prob/2
                    else: # normal scenario
                        next_s = max(0, next_s - 3)
                        matrix_normal[s][next_s] += prob/2
                elif layout[next_s] == 4: # handle type 4 trap
                    matrix_normal[s][next_s] += prob / 2
                    for i in range(0,15):
                        matrix_normal[s][i] += prob/30

                next_s = 3 + (step - 1)
                if layout[next_s] in  [0,3]: # normal or prison square
                    matrix_normal[s][next_s] += prob
                elif layout[next_s] == 1: # handle type 1 trap
                    matrix_normal[s][next_s] += prob/2
                    next_s = 0
                    matrix_normal[s][next_s] += prob/2
                elif layout[next_s] == 2: # handle type 2 trap
                    matrix_normal[s][next_s] += prob / 2
                    next_s = max(0,next_s - 3)
                    matrix_normal[s][next_s] += prob/2
                elif layout[next_s] == 4: # handle type 4 trap
                    matrix_normal[s][next_s] += prob / 2
                    for i in range(0,15):
                        matrix_normal[s][i] += prob/30
                continue

            next_s = s + step
            next_s = next_s % 15 if circle else min(14, next_s) # handle the circle scenario on the "normal" squares
            if layout[next_s] in [1,2,4]: # handle the trap scenarios on the "normal" squares
                prob /= 2
                if layout[next_s] == 1: # handle type 1 trap
                    matrix_normal[s][next_s] += prob
                    next_s = 0
                    matrix_normal[s][next_s] += prob
                    continue
                elif layout[next_s] == 2: # handle type 2 trap
                    matrix_normal[s][next_s] += prob
                    if next_s == 10:
                        next_s = 0
                    elif next_s == 11:
                        next_s = 1
                    elif next_s == 12:
                        next_s = 2
                    else:
                        next_s = max(0, next_s - 3)
                    matrix_normal[s][next_s] += prob
                    continue
                elif layout[next_s] == 4: # handle type 4 trap
                    matrix_normal[s][next_s] += prob
                    prob /= 15
                    for i in range(0,15):
                        matrix_normal[s][i] += prob
                    continue
            matrix_normal[s][next_s] += prob




        for step, prob in enumerate(P_risky): # calculate the transtition probability for the risky dice
            if s == 7 and step == 3: # if the state is 7 and the step is 3 then the next state is 14
                    next_s = 14
                    matrix_risky[s][next_s] += prob
                    continue

            elif s == 8 and step in[2,3]: # if the state is 8 and the step is 2 or 3 then the next state is 14 + step - 2
                if not circle or step == 2:
                    next_s = 14
                    matrix_risky[s][next_s] += prob
                    continue
                if circle :
                    next_s = 0
                    matrix_risky[s][next_s] += prob
                    continue
            elif s == 9 and step in [1,2,3]: # if the state is 9 and the step is 1,2 or 3 then the next state is 14 + step - 1
                if not circle or step==1:
                    next_s = 14
                    matrix_risky[s][next_s] += prob
                    continue
                if circle and step == 2:
                    next_s = 0
                    matrix_risky[s][next_s] += prob
                    continue
                if circle and step == 3: # special scenario for the circle
                    next_s = 1
                    if layout[next_s] != 0:
                        if layout[next_s] == 1:
                            next_s = 0
                            matrix_risky[s][next_s] += prob
                            continue
                        elif layout[next_s] == 2:
                            next_s = max(0, next_s - 3)
                            matrix_risky[s][next_s] += prob
                            continue
                        elif layout[next_s] == 4:
                            prob /= 15
                            for i in range(0, 15):
                                matrix_risky[s][i] += prob
                            continue

                    matrix_risky[s][next_s] += prob
                    continue

            elif s == 2 and step > 0: # handle the fast lane
                prob /= 2
                next_s = 10 + (step - 1)
                if layout[next_s] == 1: # handle type 1 trap
                    next_s = 0
                    matrix_risky[s][next_s] += prob
                elif layout[next_s] == 2: # handle type 2 trap
                    if next_s == 10:
                        next_s = 0
                    elif next_s == 11:
                        next_s = 1
                    elif next_s == 12:
                        next_s = 2
                    else:
                        next_s = max(0, next_s - 3)
                    matrix_risky[s][next_s] += prob
                elif layout[next_s] == 4: # handle type 4 trap
                    for i in range(0,15):
                        matrix_risky[s][i] += prob/15
                else:
                    matrix_risky[s,next_s] += prob
                next_s = 3 + (step - 1)
                matrix_risky[s,next_s] += prob
                continue
            next_s = s + step
            next_s = next_s % 15 if circle else min(14, next_s) # handle the circle scenario on the "normal" squares
            if layout[next_s] in [1,2,4]: # handle the trap scenarios on the "normal" squares
                if layout[next_s] == 1:
                    next_s = 0
                    matrix_risky[s][next_s] += prob
                    continue
                elif layout[next_s] == 2: # handle type 2 trap
                    if next_s == 10:
                        next_s = 0
                        matrix_risky[s][next_s] += prob
                        continue
                    elif next_s == 11:
                        next_s = 1
                        matrix_risky[s][next_s] += prob
                        continue
                    elif next_s == 12:
                        next_s = 2
                        matrix_risky[s][next_s] += prob
                        continue
                    else:
                        next_s = max(0, next_s - 3)
                        matrix_risky[s][next_s] += prob
                        continue
                elif layout[next_s] == 4: # handle type 4 trap
                    prob /= 15
                    for i in range(0,15):
                        matrix_risky[s][i] += prob
                    continue
            matrix_risky[s,next_s] += prob
            # print an array of the sum of each row






    return matrix_safe, matrix_normal, matrix_risky


def generate_arrays(n):
    # Initialize an empty list to store all the arrays
    arrays = []

    for _ in range(n):
        # Initialize a zero array of size 15
        array = np.zeros(15, dtype=int)

        # Generate 3 random indices between 1 and 13 (exclusive)
        indices = random.sample(range(1, 14), 4)

        # Assign the values 1, 2 and 3 to the randomly generated indices
        array[indices] = 1, 2, 3,4

        # Append the generated array to the list
        arrays.append(array)

    print(arrays)

    return arrays
# create a function that test the transition matrix for different layout each time with one trap of each sort
def tst_transition_matrix():
    # create a list of 100 different layouts
    layouts = generate_arrays(100)
    for array in layouts:
        print(array)
        compute_transition_matrix(array, False)
        compute_transition_matrix(array, True)

