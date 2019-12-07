import numpy as np
import random

class ReplayBuffer:
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def max_size(self):
        return self._maxsize

    def add(self, transition):
        # transition = (current_state, action, goal, reward, next_state, satisfied_goals_t, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(transition)
        else:
            self._storage[self._next_idx] = transition
        self._next_idx = int((self._next_idx + 1) % self._maxsize)

    def _encode_sample(self, batch_size):
        state_t, actions, goals, rewards, state_tp1, satisfied_goals, dones = [], [], [], [], [], [], []
        for i in batch_size:
            t = self._storage[i]

            state_t.append(np.array(t.current_state, copy=False))
            actions.append(np.array(t.action, copy=False))
            goals.append(np.array(t.goal, copy=False))
            rewards.append(t.reward)
            state_tp1.append(np.array(t.next_state, copy=False))
            satisfied_goals.append(t.satisfied_goals_t)
            dones.append(t.done)

        return np.array(state_t), np.array(actions), np.array(goals), np.array(rewards), np.array(state_tp1), np.array(satisfied_goals), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)