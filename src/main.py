import sys
import os
import importlib
import numpy as np
from neural_network import NN
from replay_buffer import ReplayBuffer
from rl import RL_AC

PATH_TO_CONF = os.path.join(os.getcwd(), 'confs')
sys.path.append(PATH_TO_CONF)
N_try = 0

def compute_sample(args):
    ep, ICS = args[0], args[1]
    # Create initial TO
    init_rand_state, init_TO_states, init_TO_controls, NSTEPS_SH, success_init_flag = rlac.create_TO_init(ep, ICS)
    if success_init_flag == 0:
        return None

    # Solve TO problem
    TO_controls, TO_states, success_flag, TO_ee_pos_arr, TO_step_cost, dVdx = TrOp.TO_Solve(init_rand_state, init_TO_states, init_TO_controls, NSTEPS_SH)
    if success_flag == 0:
        return None
    
    # Collect experiences 
    state_arr, partial_reward_to_go_arr, total_reward_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, term_arr, ep_return, RL_ee_pos_arr  = rlac.RL_Solve(TO_controls, TO_states, TO_step_cost)

    if conf.env_RL == 0:
        RL_ee_pos_arr = TO_ee_pos_arr

    return NSTEPS_SH, TO_controls, TO_ee_pos_arr, dVdx, state_arr.tolist(), partial_reward_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, term_arr, ep_return, RL_ee_pos_arr

if __name__ == '__main__':
    conf = importlib.import_module('iiwa_conf')
    env = getattr(conf, 'Env')(conf)
    nn = NN(env, conf)
    buffer = ReplayBuffer(conf)
    rlac = RL_AC(env, nn, conf, N_try)

    rlac.setup_model()

    # Initialize arrays to store the reward history of each episode and the average reward history of last 100 episodes
    ep_arr_idx = 0
    ep_reward_arr = np.zeros(conf.NEPISODES-ep_arr_idx)*np.nan

    for ep in range(conf.NLOOPS):
        init_rand_state = env.reset_batch(conf.EP_UPDATE)

        tmp = []
        for i in range(conf.EP_UPDATE):
            # make samples and add to buffer
            result = compute_sample((ep, init_rand_state[i, :]))
            tmp.append(result)
        tmp = [x for x in tmp if x is not None]
        NSTEPS_SH, TO_controls, ee_pos_arr_TO, dVdx, state_arr, partial_reward_to_go_arr, state_next_rollout_arr, done_arr, rwrd_arr, term_arr, ep_return, ee_pos_arr_RL = zip(*tmp)
        buffer.add(state_arr, partial_reward_to_go_arr, state_next_rollout_arr, dVdx, done_arr, term_arr)

    

