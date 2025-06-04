import numpy as np
import torch

class ReplayBuffer(object):
    def __init__(self, conf):
        self.conf = conf
        self.storage_mat = np.zeros((conf.REPLAY_SIZE, conf.state_dim + 1 + conf.state_dim + 1))
        self.next_idx = 0
        self.full = False
        self.capacity = conf.REPLAY_SIZE
        self.batch_size = conf.BATCH_SIZE
        self.state_dim = conf.state_dim

    def add(self, obses_t, rewards, obses_t1, dones):
        data = self.concatenate_sample(obses_t, rewards, obses_t1, dones)
        end_idx = self.next_idx + len(data)

        if end_idx > self.capacity:
            overflow = end_idx - self.capacity
            self.storage_mat[self.next_idx:,:] = data[:self.capacity - self.next_idx]
            self.storage_mat[:overflow] = data[self.capacity - self.next_idx:]
            self.full = True
        else:
            self.storage_mat[self.next_idx:end_idx] = data

        self.next_idx = end_idx % self.capacity

    def sample(self):
        # Select indexes of the batch elements
        max_idx = self.capacity if self.full else self.next_idx
        idxes = np.random.randint(0, max_idx, size=self.batch_size)

        obses_t = self.storage_mat[idxes, :self.state_dim]
        rewards = self.storage_mat[idxes, self.state_dim:self.state_dim+1]
        obses_t1 = self.storage_mat[idxes, self.state_dim+1:self.state_dim*2+1]
        dones = self.storage_mat[idxes, self.state_dim*2+1:self.state_dim*3+1]
        weights = np.ones((self.conf.BATCH_SIZE,1))

        # Convert the sample in tensor
        obses_t, rewards, obses_t1, dones, weights = self.convert_sample_to_tensor(
            obses_t, rewards, obses_t1, dones, weights)
        return obses_t, rewards, obses_t1, dones, weights

    def concatenate_sample(self, obses_t, rewards, obses_t1, dones):
        obses_t  = np.concatenate(obses_t, axis=0)
        rewards  = np.concatenate(rewards, axis=0).reshape(-1, 1)
        obses_t1 = np.concatenate(obses_t1, axis=0)
        dones    = np.concatenate(dones, axis=0).reshape(-1, 1)
        return np.concatenate([obses_t, rewards, obses_t1, dones], axis=1) 

    def convert_sample_to_tensor(self, obses_t, rewards, obses_t1, dones, weights):
        obses_t = torch.tensor(obses_t, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        obses_t1 = torch.tensor(obses_t1, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)
        
        return obses_t, rewards, obses_t1, dones, weights