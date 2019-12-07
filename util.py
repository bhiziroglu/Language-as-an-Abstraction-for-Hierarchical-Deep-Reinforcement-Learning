import numpy as np
import torch
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
import networks
import random
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Transition:
    def __init__(self, current_state, action, goal, reward, next_state, satisfied_goals_t, done):
        self.current_state = current_state
        self.action = action
        self.goal = goal
        self.reward = reward
        self.next_state = next_state
        self.satisfied_goals_t = satisfied_goals_t
        self.done = done

def get_state_based_representation(observation, ghat, f1_model):
    '''
    Computation graph of the state-based low level policy.
    '''

    if len(observation.shape) == 2:
        observation = np.expand_dims(observation, 0)

    observation = torch.Tensor(observation).to(DEVICE)

    # Create Z Matrix
    data = []
    for i in range(observation.shape[0]):
        for j in range(observation.shape[1]):
            for k in range(observation.shape[1]):
                data.append(torch.cat((observation[i, j, :], observation[i, k, :]), 0).to(DEVICE))
    output = f1_model(torch.stack(data))
    Z_matrix = output.view(observation.shape[0], observation.shape[1], observation.shape[1], -1)

    # Check for batch
    if len(ghat.shape) == 1:
        # Get Ghat
        ghat = ghat.unsqueeze(0)

    batch_size = len(Z_matrix)
    dim_1 = len(Z_matrix[0])

    # Create p matrix (Figure 8 top right matrix)
    w_matrix = torch.stack([torch.dot(z_vec, ghat[idx]) for idx, batch in enumerate(Z_matrix) for row in batch for z_vec in row])
    p_matrix = F.softmax(w_matrix.view(batch_size, -1), dim=1)
    p_matrix = p_matrix.view(-1, dim_1, dim_1)  

    # Create z vector
    z_vector = [[[0.0 for _ in range(5)] for _ in range(5)] for batch in observation]

    for i in range(observation.shape[0]):
        for j in range(observation.shape[1]):
            for k in range(observation.shape[1]):
                z_vector[i][j][k] = torch.sum(p_matrix[i][j][k] * Z_matrix[i][j][k])
    
    zhat = torch.stack([torch.stack([torch.sum(torch.stack(rows)) for rows in batch]) for batch in z_vector])

    # Each o is concatenated with g and z
    state_rep = [[0.0 for _ in range(5)] for batch in observation]
    for i in range(observation.shape[0]):
        for j in range(observation.shape[1]):
            current_o = observation[i, j, :]
            state_rep[i][j] = torch.cat([current_o, ghat[i], zhat[i]],0)

    out = torch.stack([torch.stack(batch) for batch in state_rep])

    return out

def future_instruction_relabeling_strategy(trajectory, t, k, discount_factor):
    '''
    Future Instruction Relabeling Strategy
    (Algorithm 4 in the paper)
    '''
    if len(trajectory) - 1 == t: 
        return []
    deltas = []
    for _ in range(k):
        future = random.randint(t+1, len(trajectory)-1)
        transition = trajectory[future]
        if transition.satisfied_goals_t:
            index = random.randint(0, len(transition.satisfied_goals_t)-1)
            goal_prime = transition.satisfied_goals_t[index]
            reward_prime = transition.reward * pow(discount_factor, future-t)
            deltas.append([goal_prime, reward_prime])
    return deltas
