import sys
import os
import importlib
import numpy as np
from neural_network import NN
from replay_buffer import ReplayBuffer
from rl import RL_AC
from opt_control.traj_opt import TO

# -----Sample computation function-----------------------------------------------------------------
def compute_sample(args):
    ep, init_state, env = args
    init_state, init_states, init_controls, success = rlac.create_TO_init(ep, init_state)
    if not success: return None
    print("after create TO init", env.ee(init_states[-1]))
    TO_states, TO_controls = TrOp.TO_Solve(init_state, init_states, init_controls)
    print("after TO solve", env.ee(TO_states[-1]))
    RL_states, partial_rtg, next_states, done, rewards = rlac.RL_Solve(TO_controls, TO_states)
    print("after RL solve", env.ee(RL_states[-1]))
    return RL_states.tolist(), partial_rtg, next_states, done, rewards


if __name__ == '__main__':
    # -----Initialization--------------------------------------------------------------------------
    # set up conf
    PATH_TO_CONF = os.path.join(os.getcwd(), 'confs')
    sys.path.append(PATH_TO_CONF)
    conf = importlib.import_module('iiwa_conf')

    # initialze env, nn, buffer, TO, and rl
    env = getattr(conf, 'Env')(conf)
    nn = NN(env, conf)
    buffer = ReplayBuffer(conf)
    TrOp = TO(env, conf)
    rlac = RL_AC(env, nn, conf, 0)
    rlac.setup_model()

    # initialize episode reward arrays
    ep_arr_idx = 0
    ep_reward_arr = np.zeros(conf.NEPISODES-ep_arr_idx)*np.nan
    update_step_counter = 0

    # -----Episode loop----------------------------------------------------------------------------
    for ep in range(conf.NLOOPS):
        # collect samples
        init_rand_state = env.reset_batch(conf.EP_UPDATE)
        samples = [compute_sample((ep, init_rand_state[i, :], env)) for i in range(conf.EP_UPDATE)]
        valid_samples = [sample for sample in samples if sample]
        
        # add samples to replay buffer
        print(f"Compute_sample {len(valid_samples)}/{conf.EP_UPDATE} success")
        states, partial_rewards, state_nexts, dones, rewards = zip(*valid_samples)
        buffer.add(states, partial_rewards, state_nexts, dones)

        # Update nns and record rewards
        update_step_counter = rlac.learn_and_update(update_step_counter, buffer, ep)
        ep_reward_arr[ep_arr_idx : ep_arr_idx + num_success] = rewards
        ep_arr_idx += num_success

        for i in range(len(valid_samples)):
            print("Episode  {}  --->   Return = {}".format(ep*len(tmp) + i, rewards[i]))

        if update_step_counter > conf.NUPDATES:
            break
    
    rlac.RL_save_weights()

    

