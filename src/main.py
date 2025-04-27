import sys
import os
import importlib
import numpy as np
from neural_network import NN
from replay_buffer import ReplayBuffer
from rl import RL_AC
from opt_control.traj_opt import TO

PATH_TO_CONF = os.path.join(os.getcwd(), 'confs')
sys.path.append(PATH_TO_CONF)
N_try = 0

def compute_sample(args):
    ep, init_state, env = args
    init_state, init_states, init_controls, success = rlac.create_TO_init(ep, init_state)
    if not success: return None
    print("after create TO init", env.ee(init_states[-1]))
    TO_states, TO_controls, ee_pos = TrOp.TO_Solve(init_state, init_states, init_controls)
    print("after TO solve", env.ee(TO_states[-1]))
    print("after TO solve", ee_pos[-1])
    states, partial_rtg, next_states, done, rewards, rl_ee_pos = rlac.RL_Solve(TO_controls, TO_states)
    print("after RL solve", env.ee(states[-1]))
    return TO_controls, ee_pos, states.tolist(), partial_rtg, next_states, done, rewards, rl_ee_pos

if __name__ == '__main__':
    conf = importlib.import_module('iiwa_conf')
    env = getattr(conf, 'Env')(conf)
    nn = NN(env, conf)
    buffer = ReplayBuffer(conf)
    rlac = RL_AC(env, nn, conf, N_try)
    TrOp = TO(env, conf)

    rlac.setup_model()

    # Initialize arrays to store the reward history of each episode and the average reward history of last 100 episodes
    ep_arr_idx = 0
    ep_reward_arr = np.zeros(conf.NEPISODES-ep_arr_idx)*np.nan
    update_step_counter = 0

    for ep in range(conf.NLOOPS):
        init_rand_state = env.reset_batch(conf.EP_UPDATE)

        tmp = []
        for i in range(conf.EP_UPDATE):
            # make samples and add to buffer
            result = compute_sample((ep, init_rand_state[i, :], env))
            tmp.append(result)
        tmp = [x for x in tmp if x is not None]
        print(f"Compute_sample {len(tmp)}/{conf.EP_UPDATE} success")
        states, partial_rewards, state_nexts, dones, rewards = zip(*tmp)
        buffer.add(states, partial_rewards, state_nexts, dones)


        # Update NNs
        update_step_counter = rlac.learn_and_update(update_step_counter, buffer, ep)
        ep_reward_arr[ep_arr_idx:ep_arr_idx+len(tmp)] = rewards
        ep_arr_idx += len(tmp)

        for i in range(len(tmp)):
            print("Episode  {}  --->   Return = {}".format(ep*len(tmp) + i, rewards[i]))

        if update_step_counter > conf.NUPDATES:
            break
    
    rlac.RL_save_weights()

    

