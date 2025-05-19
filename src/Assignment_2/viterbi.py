import numpy as np


def viterbi_algorithm(sequence, initial_probs, transition_probs, emission_probs):
    """
    Viterbi Algorithm to calculate the most probable sequence of hidden states for a given observed sequence.

    Parameters:
    sequence (list): The observed sequence of emissions 
    initial_probs (dict): The initial probabilities for each state 
    transition_probs (dict): The transition probabilities between states 
    emission_probs (dict): The emission probabilities for each state and observed symbol 

    Returns:
    tuple: (most probable state sequence, probability of that sequence)
    """

    states = list(initial_probs.keys())
    n = len(sequence)  # Length of the sequence
    dp = {state: [0] * n for state in states}  # Store delta values
    psi = {state: [None] * n for state in states}  # Store back pointers

    # Initialization (t=1)
    for state in states:
        dp[state][0] = initial_probs[state] * \
            emission_probs[state][sequence[0]]
        psi[state][0] = None  # No previous state at t=1

    # Recursion (t=2 to t=n)
    for t in range(1, n):
        for state in states:
            max_prob = -1  # Initialize a very low number for the max probability
            max_state = None  # Store the best previous state

            for prev_state in states:
                prob = dp[prev_state][t-1] * transition_probs[prev_state][state] * \
                    emission_probs[state][sequence[t]]
                if prob > max_prob:
                    max_prob = prob
                    max_state = prev_state

            dp[state][t] = max_prob
            psi[state][t] = max_state

    # Termination: Find the max probability at the last time step
    final_state = max(states, key=lambda state: dp[state][n-1])
    final_prob = dp[final_state][n-1]

    # Backtracking to find the most probable state sequence
    most_probable_sequence = [None] * n
    most_probable_sequence[n-1] = final_state
    for t in range(n-2, -1, -1):
        most_probable_sequence[t] = psi[most_probable_sequence[t+1]][t+1]

    return most_probable_sequence, final_prob

if __name__ == "__main__":
    
    initial_probs = {'Exon': 0.5, 'Intron': 0.5}
    transition_probs = {
        'Exon': {'Exon': 0.9, 'Intron': 0.1},
        'Intron': {'Exon': 0.2, 'Intron': 0.8}
    }
    emission_probs = {
        'Exon': {'A': 0.25, 'U': 0.25, 'G': 0.25, 'C': 0.25},
        'Intron': {'A': 0.4, 'U': 0.4, 'G': 0.05, 'C': 0.15}
    }
    sequence = "AUUAU"

    # Apply Viterbi algorithm
    state_sequence, sequence_prob = viterbi_algorithm(
        sequence, initial_probs, transition_probs, emission_probs)

    # Print the result
    print("Most probable state sequence:", state_sequence)
    print("Probability of the sequence:", sequence_prob)

    
