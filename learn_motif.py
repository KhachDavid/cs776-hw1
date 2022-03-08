import argparse
import sys
import numpy as np
import math
from util import *
# You can choose to write classes in other python files
# and import them here.
from PWM import PWMList


ALPHABET = ["A", "C", "G", "T"]                             # List of characters in the alphabet  
alphabet_size = 4                                           # Number of characters
FIXED_THRESHOLD = 0.001                                     # Threshold to stop the EM algorithm and declare that it converged
pi = 0.7                                                    # Initial probability for the model that matches with the base
pi_not_matching = round((1 - pi) / (alphabet_size - 1), 3)  # Initial probability for the model that does not match with the base


# This is the main function provided for you.
# Define additional functions to implement MEME
def main(args):
    # Parse input arguments
    # It is generally good practice to validate the input arguments, e.g.,
    # verify that the input and output filenames are provided
    seq_file_path = args.sequences_filename
    W = args.width
    model_file_path = args.model
    position_file_path = args.positions
    subseq_file_path = args.subseqs

    # Read the sequences from the file
    # You can use the functions in learn_motif.py
    try:
        sequences = read_sequences(seq_file_path)
    except IOError:
        print("Error: could not read the sequences file")
        sys.exit(1)

    """
    For every distinct subsequences of length W in the training set
     - derive an initial p matrix from this subsequence
     - run EM for 1 iteration
    Choose motif model with highest likelihood
    Run EM to convergence for that motdel 
    """

    """
    Checklist of 9 things:
    1. Dna sequences: sub_sequences
    2. L: number of bases in each sequence
    3. W: number of bases in the motif
    4. pseudocount: pseudocount for maximum likelihood
    5. M: possible starting positions of the motif in each sequence
    6. unique_motifs: unique motifs in the training set
    7. unique_motifs_count: number of unique motifs in the training set
    8. z_matrix: z matrix showing the initial probability of each motif
    9. pwm_dict: pwm dictionary for each motif, where a motif is mapped to a 2d numpy array of probabilities
    10. pi = 0.7
    """
    dna_sequences = sequences
    L = len(dna_sequences[0])
    W = W
    pseudocount = 1
    M = L - W + 1
    unique_motifs = get_unique_motifs(dna_sequences, W)

    # initialize z_matrix with len(dna_sequences) rows and W columns and each value is equal to 1 / alphabet_size
    z_matrix = np.ones((len(dna_sequences), M)) / M

    # initialize pwm_dict with len(unique_motifs) rows each mapping to a 2d numpy array of probabilities
    pwm_dict = PWMList(unique_motifs, W, ALPHABET, pi, pi_not_matching)

    base_counts = [
        count_bases(dna_sequences, 'A'),
        count_bases(dna_sequences, 'C'),
        count_bases(dna_sequences, 'G'),
        count_bases(dna_sequences, 'T')
    ]

    # initialize the variable that holds the best motif so far
    motif_to_consider = [np.NINF, np.NINF, np.NINF, np.NINF]

    for key in pwm_dict.keys():
        current_motif = key

        print("Running EM for motif: {}".format(current_motif))

        #  run the e-step
        print("Running E-step for motif: {}".format(current_motif))
        z_matrix = e_step(
            dna_sequences, pwm_dict.pwm[key], z_matrix, M, W, current_motif)

        # run the m-step
        print("Running M-step for motif: {}".format(current_motif))
        pwm_dict.pwm[key] = m_step(
            dna_sequences, z_matrix, pwm_dict.pwm[key], M, W, pseudocount, base_counts)

        # compute the log likelihood
        print("Computing log likelihood for motif: {}".format(current_motif))
        log_probability = compute_log_likelihood(
            dna_sequences, pwm_dict.pwm[key], z_matrix, M, W, current_motif, pseudocount, base_counts)

        print("Log likelihood for motif: {}".format(log_probability))

        motif_to_consider = [log_probability, current_motif, z_matrix, pwm_dict.pwm[key]
                             ] if log_probability > motif_to_consider[0] else motif_to_consider
        print(
            f"Current best log likelihood is: {motif_to_consider[0]} belonging to motif: {motif_to_consider[1]}")


    print("After one iteration of EM over each unique motif, these are the findings")
    print(
        f'The log likelihood for {motif_to_consider[1]} is the highest with: {motif_to_consider[0]}')
    print("The pwm for the most likely motif is:")
    print(print_pwm_with_labels(motif_to_consider[3], ALPHABET))

    current_log_likelihood = motif_to_consider[0]
    current_pwm = motif_to_consider[3]
    number_of_times_ran = 1
    current_motif = motif_to_consider[1]

    while True:
        number_of_times_ran += 1

        z_matrix = e_step(
            dna_sequences, current_pwm, z_matrix, M, W, current_motif)
        current_pwm = m_step(dna_sequences, z_matrix,
                             current_pwm, M, W, pseudocount, base_counts)

        print("\nStep {}:".format(number_of_times_ran))
        print("The pwm is:")
        print(print_pwm_with_labels(current_pwm, ALPHABET))
        log_probability = compute_log_likelihood(
            dna_sequences, current_pwm, z_matrix, M, W, current_motif, pseudocount, base_counts)

        print("The log likelihood is:")
        print(log_probability)

        if np.abs(current_log_likelihood - log_probability) < FIXED_THRESHOLD:
            write_model(print_pwm_with_labels(current_pwm, ALPHABET), model_file_path)
            positions = write_positions(z_matrix, position_file_path)
            write_subsequences(positions, dna_sequences, W, subseq_file_path)
            break
        else:
            current_log_likelihood = log_probability


def e_step(sequences, pwm, z_matrix, M, W, current_motif):
    """
    For each sequence in the training set
        - for each position in the sequence
            - for each motif in the motif set
                - calculate the probability of the sequence at this position
                - update the z_matrix
    """
    for i in range(len(sequences)):
        for j in range(M):
            z_matrix[i][j] = calculate_probability(
                sequences[i], j, pwm, M, W, current_motif)

        # normalize z_matrix[i]
        z_matrix[i] = z_matrix[i] / np.sum(z_matrix[i])

    return z_matrix


def m_step(sequences, z_matrix, pwm, M, W, pseudocount, base_counts):
    """
    :param sequences: a list of sequences
    :param z_matrix: a 2d numpy array
    :param model: a 2d numpy array
    :param W: the width of the motif
    :return: a 2d numpy array
    """
    pwm = np.zeros((len(ALPHABET), W + 1))

    for i in range(len(sequences)):
        current_sequence = sequences[i]
        for j in range(M):
            this_motif = current_sequence[j:j+W]
            for k in range(W):
                pwm[ALPHABET.index(
                    this_motif[k])][k + 1] += z_matrix[i][j]

    for index, character in enumerate(ALPHABET):
        pwm[ALPHABET.index(character)][0] = base_counts[index] - \
            np.sum(pwm[ALPHABET.index(character)][1:])

    for i in range(W + 1):
        pwm[:, i] = (pwm[:, i] + pseudocount) / \
            (np.sum(pwm[:, i]) + pseudocount * len(ALPHABET))

    return pwm


def compute_log_likelihood(sequences, pwm, z_matrix, M, W, current_motif, pseudocount, base_counts):
    """
    :param sequences: a list of sequences
    :param z_matrix: a 2d numpy array
    :param model: a 2d numpy array
    :param W: the width of the motif
    :return: a float
    """
    log_likelihood = 0

    for sequence in sequences:
        sequence_prob = 0
        for i in range(M):
            sequence_prob += compute_sequence_probability(
                sequence, pwm, i, ALPHABET, W)
        log_likelihood += np.log(sequence_prob/len(sequences))

    return log_likelihood


def count_corresponding_z(z_matrix, base, k, W, M, sequences):
    """
    :param z_matrix: a 2d numpy array
    :param base: a base
    :param k: a position
    :param sequences: a list of sequences
    :return: Return the n_base,k value
    The number of times the base occurs at position k in all the motifs in the set
    """
    sum_of_z_for_current_motif = 0
    for seq_index, row in enumerate(z_matrix):
        current_sequence = sequences[seq_index]
        current_motif = current_sequence[k:k + W]
        for index, character in enumerate(current_motif):
            if character == base:
                sum_of_z_for_current_motif = z_matrix[seq_index][k]
    return sum_of_z_for_current_motif


def calculate_probability(sequence, position, pwm, M, W, current_motif):
    """
    :param sequence: a sequence
    :param position: a position in the sequence where the motif starts
    :param pwm: a pwm
    :return: the probability of the motif starting at this position
    """

    starting_position = position
    ending_position = position + W

    p_values = []
    for index, character in enumerate(sequence):
        if index >= starting_position and index < ending_position:
            value = pwm[ALPHABET.index(
                character)][index - starting_position + 1]
            p_values.append(value)
        else:
            value = pwm[ALPHABET.index(character)][0]
            p_values.append(value)

    probability = (np.prod(p_values))
    return probability


def get_unique_motifs(sequences, W):
    """
    Return a list of all unique motifs of length W in the sequences
    """
    unique_motifs = []
    for sequence in sequences:
        for i in range(len(sequence) - W + 1):
            unique_motifs.append(sequence[i:i+W])
    return list(set(unique_motifs))


def read_sequences(seq_file_path):
    """
    :param seq_file_path:
    :return: a list of sequences
    """
    sequences = []
    with open(seq_file_path, 'r') as f:
        for line in f:
            sequences.append(line.strip())
    return sequences


def write_model(model, output_file_path):
    """
    :param model: a 2d numpy array
    :param output_file_path: a string
    """
    
    # write the model to a file
    with open(output_file_path, 'w') as f:
        # model is a string
        f.write(model)


def write_positions(z_matrix, output_file_path):
    """
    :param z_matrix: a 2d numpy array
    :param output_file_path: a string
    """
    positions = []

    # loop over the z_matrix and find the max value for each row
    for index, row in enumerate(z_matrix):
        max_index = -1
        max_value = np.NINF
        for index, value in enumerate(row):
            if value >= max_value:
                max_value = value
                max_index = index
        positions.append(max_index)

    with open(output_file_path, 'w') as f:
        for position in positions:
            f.write(str(position) + '\n')
    
    return positions


def write_subsequences(positions, sequences, W, output_file_path):
    """
    :param positions: a list of positions
    :param sequences: a list of sequences
    :param output_file_path: a string
    """
    with open(output_file_path, 'w') as f:
        for index, position in enumerate(positions):
            f.write(sequences[index][position:position+W] + '\n')

# Note: this syntax checks if the Python file is being run as the main program
# and will not execute if the module is imported into a different module
if __name__ == "__main__":
    # Note: this example shows named command line arguments.  See the argparse
    # documentation for positional arguments and other examples.
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sequences_filename',
                        help='sequences file path.',
                        type=str)
    parser.add_argument('--width',
                        help='width of the motif.',
                        type=int,
                        default=6)
    parser.add_argument('--model',
                        help='model output file path.',
                        type=str,
                        default='model.txt')
    parser.add_argument('--positions',
                        help='position output file path.',
                        type=str,
                        default='positions.txt')
    parser.add_argument('--subseqs',
                        help='subsequence output file path.',
                        type=str,
                        default='subseqs.txt')

    args = parser.parse_args()
    # Note: this simply calls the main function above, which we could have
    # given any name
    main(args)
