import numpy as np
import torch

class ReplayBuffer(object):
    def __init__(self, conf):
        '''
        :input conf :                           (Configuration file)

            :param REPLAY_SIZE :                (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories are dropped
            :param BATCH_SIZE :                 (int) Size of the mini-batch 
            :param nb_state :                   (int) State size (robot state size + 1)
        '''

        self.conf = conf
        self.storage_mat = np.zeros((conf.REPLAY_SIZE, conf.nb_state + 1 + conf.nb_state + 1))
        self.next_idx = 0
        self.full = 0
        self.exp_counter = np.zeros(conf.REPLAY_SIZE)

    def add(self, obses_t, rewards, obses_t1, dones):
        ''' Add transitions to the buffer '''
        data = self.concatenate_sample(obses_t, rewards, obses_t1, dones)

        if len(data) + self.next_idx > self.conf.REPLAY_SIZE:
            self.storage_mat[self.next_idx:,:] = data[:self.conf.REPLAY_SIZE-self.next_idx,:]
            self.storage_mat[:self.next_idx+len(data)-self.conf.REPLAY_SIZE,:] = data[self.conf.REPLAY_SIZE-self.next_idx:,:]
            self.full = 1
        else:
            self.storage_mat[self.next_idx:self.next_idx+len(data),:] = data

        self.next_idx = (self.next_idx + len(data)) % self.conf.REPLAY_SIZE

    def sample(self):
        ''' Sample a batch of transitions '''
        # Select indexes of the batch elements
        if self.full:
            max_idx = self.conf.REPLAY_SIZE
        else:
            max_idx = self.next_idx
        idxes = np.random.randint(0, max_idx, size=self.conf.BATCH_SIZE) 

        obses_t = self.storage_mat[idxes, :self.conf.nb_state]
        rewards = self.storage_mat[idxes, self.conf.nb_state:self.conf.nb_state+1]
        obses_t1 = self.storage_mat[idxes, self.conf.nb_state+1:self.conf.nb_state*2+1]
        dones = self.storage_mat[idxes, self.conf.nb_state*2+1:self.conf.nb_state*3+1]

        # Priorities not used
        weights = np.ones((self.conf.BATCH_SIZE,1))
        batch_idxes = None

        # Convert the sample in tensor
        obses_t, rewards, obses_t1, dones, weights = self.convert_sample_to_tensor(obses_t, rewards, obses_t1, dones, weights)
        
        return obses_t, rewards, obses_t1, dones, weights, batch_idxes

    def concatenate_sample(self, obses_t, rewards, obses_t1, dones):
        ''' Convert batch of transitions into a tensor '''
        obses_t = np.concatenate(obses_t, axis=0)
        rewards = np.concatenate(rewards, axis=0)                                 
        obses_t1 = np.concatenate(obses_t1, axis=0)
        dones = np.concatenate(dones, axis=0)
        return np.concatenate((obses_t, rewards.reshape(-1,1), obses_t1, dones.reshape(-1,1)),axis=1)
    

    def convert_sample_to_tensor(self, obses_t, rewards, obses_t1, dones, weights):
        ''' Convert batch of transitions into a tensor using PyTorch '''
        obses_t = torch.tensor(obses_t, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        obses_t1 = torch.tensor(obses_t1, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)
        
        return obses_t, rewards, obses_t1, dones, weights