import numpy as np

def compute_sequence_probability(sequence, pwm, starting_position, alphabet, W):
    probability = 1
    for j in range(len(sequence)):
        if j >= starting_position and j < starting_position + W:
            probability *= pwm[alphabet.index(sequence[j])][j - starting_position + 1]
        else:
            probability *= pwm[alphabet.index(sequence[j])][0]
    return probability


def count_bases(sequences, character): 
    count = 0
    for sequence in sequences:
        count += sequence.count(character)
    return count


def count_character(z_matrix):
    count = 0
    for row in z_matrix:
        # find the sum of all the elements
        count += np.sum(row)
    return count


def print_matrix(mtrx):
    rtrn = ""
    for row in mtrx:
        for column in row:
            rtrn += '| ' + str(column) + ' '
        rtrn += '|\n'
    return rtrn


def print_pwm_with_labels(pwm, alphabet):

    rtrn = ""
    for index, row in enumerate(pwm):
        rtrn += alphabet[index] + " | "
        for element in row:
            # round element to 3 decimal places
            num = str(round(element, 3))
            
            # add necessary amount of zeros if length of num is less than 3
            rtrn += str(round(element, 3)) + " "

        rtrn += "\n"
    rtrn += "Note 1: PWM is normalized to 1.0\n"
    rtrn += "Note 2: The first column represents the background probability\n"
    return rtrn
