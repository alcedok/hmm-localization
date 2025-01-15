'''
Self-Localization code using Hidden Markov Model (HMM)
'''

import numpy as np

def init_probabilities(num_states):
    '''We do not assume any bias on initial position, so we initialize with a random distribution'''
    random_init = np.random.random_sample(size=(num_states,))
    # normalize so all add to one
    random_init_normalized = random_init / np.sum(random_init)
    assert np.allclose(np.sum(random_init_normalized), 1), 'Initial probabilities do not sum to one.'
    return random_init_normalized

def get_state_probabilities(observation, transition_matrix, observation_matrix, observation_history=[]):
    '''
    Localization must return a numpy array of shape (Z,) corresponding to the probability distribution over states
    
    Computes state probabilities using Hidden Markov Model (HMM) for localization.
    
    Parameters:
        observation: numpy array of shape (4,), the current observation
        transition_matrix: numpy array of shape (N*N, N*N), transition probabilities
        observation_matrix: numpy array of shape (N*N, Z), observation probabilities
        observation_history: list of observations (numpy arrays)
    
    Returns:
        numpy array of shape (N*N,) representing state probabilities
    '''
    N = int(np.sqrt(transition_matrix.shape[0]))  # Extract grid size (N x N)
    num_states = N * N  # Total number of states
    num_observations = observation_matrix.shape[1]  # Total number of unique observations (Z)
    
    # Initial uniform belief if there's no history
    belief = np.full(num_states, 1 / num_states)
    print(belief)
    # Process the observation history
    for obs in observation_history:
        # Prediction step
        belief = np.dot(transition_matrix.T, belief)
        
        # Correction step for historical observations
        obs_index = int("".join(map(str, obs)), 2)  # Convert binary vector to integer index
        belief = observation_matrix[:, obs_index] * belief
        
        # Normalize the belief
        belief /= np.sum(belief)

    # Incorporate the latest observation
    obs_index = int("".join(map(str, observation)), 2)  # Convert binary vector to integer index
    belief = observation_matrix[:, obs_index] * belief
    belief /= np.sum(belief)  # Normalize again

    return belief  # Return as a flat array of shape (N*N,)