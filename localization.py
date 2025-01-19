'''
Self-Localization code using Hidden Markov Model (HMM)
'''

import numpy as np
from typing import Literal

def init_probabilities(num_states, mode: Literal['uniform', 'zero'] = 'zero'):
    '''
    We do not assume any bias on initial position.
    Supported initialization modes: 
    - uniform: unifrom probability across all states
    - zero: all probabilities start at zero
    '''
    if mode == 'uniform':
        random_init = np.random.random_sample(size=(num_states,))
        # normalize so all add to one
        init_probs = random_init / np.sum(random_init)
        assert np.allclose(np.sum(init_probs), 1), 'Initial probabilities do not sum to one.'
    elif mode == 'zero':
        init_probs = np.zeros(shape=(num_states, ))
    else: 
        raise NotImplementedError('mode {} is not currently supported.'.format(mode))
    return init_probs

def get_smoothed_state_probabilities(observation, transition_matrix, observation_matrix, observation_history=[], eps=1e-10):
    '''
    Computes smoothed state probabilities using the forward-backward algorithm (filtering + smoothing)
    
    inputs:
        observation: numpy array of shape (4,)
        transition_matrix: numpy array of shape (N*N, N*N)
        observation_matrix: numpy array of shape (N*N, Z)
        observation_history: list of observations (numpy arrays) prior to the current observation
        eps: small constant added to avoid division by zero in normalization
    
    outputs:
        numpy array of shape (len(observation_history)+1, N*N), smoothed probabilities for each state at each time step
    '''
    import numpy as np

    N = int(np.sqrt(transition_matrix.shape[0]))
    num_states = N * N
    # all time steps including the current observation
    T = len(observation_history) + 1  

    # init alpha (forward probs), beta (backward probs) and smoothed
    alpha = np.zeros((T, num_states))
    beta = np.zeros((T, num_states))
    smoothed_probs = np.zeros((T, num_states))

    # init uniform belief
    alpha[0] = np.full(num_states, 1 / num_states)

    # forward pass
    for t, obs in enumerate(observation_history):
        obs_index = int("".join(map(str, obs)), 2)  # convert binary vector to integer index
        # prediction step
        alpha[t + 1] = np.dot(transition_matrix.T, alpha[t])
        # correction step
        alpha[t + 1] *= observation_matrix[:, obs_index]
        alpha[t + 1] += eps  # add small number to avoid dividing by zero 
        # normalize
        alpha[t + 1] /= np.sum(alpha[t + 1])

    # include the current observation into forward pass
    obs_index = int("".join(map(str, observation)), 2)
    alpha[-1] = np.dot(transition_matrix.T, alpha[-2])
    alpha[-1] *= observation_matrix[:, obs_index]
    alpha[-1] += eps  # add small number to avoid dividing by zero 
    alpha[-1] /= np.sum(alpha[-1])

    # backward pass
    beta[-1] = np.ones(num_states)  # initialize backward probs at the final time step
    # iterate backwards, skip the last 
    for t in range(T - 2, -1, -1):
        obs_index = int("".join(map(str, observation_history[t])), 2) if t < T - 1 else obs_index
        beta[t] = np.dot(transition_matrix, observation_matrix[:, obs_index] * beta[t + 1])
        beta[t] += eps  # add small number to avoid dividing by zero 
        beta[t] /= np.sum(beta[t])

    # combine alpha and beta to compute smoothed probs
    for t in range(T):
        smoothed_probs[t] = alpha[t] * beta[t]
        smoothed_probs[t] += eps # add small number to avoid dividing by zero 
        # normalize
        smoothed_probs[t] /= np.sum(smoothed_probs[t])

    return smoothed_probs[-1]

def get_state_probabilities(observation, transition_matrix, observation_matrix, observation_history=[], eps=1e-10):
    '''    
    Filtering process (forward only); estimative state given the current observations
    
    inputs:
        observation: numpy array of shape (4,)
        transition_matrix: numpy array of shape (N*N, N*N)
        observation_matrix: numpy array of shape (N*N, Z)
        observation_history: list of observations (numpy arrays)
    
    outputs:
        state probabilities as numpy array of shape (N*N,)
    '''
    N = int(np.sqrt(transition_matrix.shape[0])) 
    num_states = N * N 
    
    # init uniform belief if there's no history
    belief = np.full(num_states, 1 / num_states)
    
    # for each item in observation history
    for obs in observation_history:
        # prediction step
        belief = np.dot(transition_matrix.T, belief)
        
        # correction step for historical observations
        obs_index = int("".join(map(str, obs)), 2)  # Convert binary vector to integer index
        belief = observation_matrix[:, obs_index] * belief
        belief += eps # add small number to avoid dividing by zero 

        # normalize
        belief /= np.sum(belief)

    # include the latest observation
    obs_index = int("".join(map(str, observation)), 2)  # Convert binary vector to integer index
    belief = observation_matrix[:, obs_index] * belief
    belief += eps # add small number to avoid dividing by zero 
    belief /= np.sum(belief)  # normalize again

    return belief