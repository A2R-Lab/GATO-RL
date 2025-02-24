import sys
import importlib
import numpy as np
from neural_network import NN
from replay_buffer import ReplayBuffer
from rl import RL_AC

PATH_TO_CONF = '/Users/seyoungree/GATO-RL/confs'
sys.path.append(PATH_TO_CONF)
N_try = 0

if __name__ == '__main__':
    conf = importlib.import_module('iiwa_conf')
    env = getattr(conf, 'Env')(conf)
    nn = NN(env, conf)
    buffer = ReplayBuffer(conf)
    RLAC = RL_AC(env, nn, conf, N_try)

    RLAC.setup_model()
