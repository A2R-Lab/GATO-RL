import sys
import os
import importlib
import numpy as np
from neural_network import ActorCriticNet
from replay_buffer import ReplayBuffer
from rl_trainer import RLTrainer
from opt.traj_opt import TrajOpt

# NOTE: Currently broken from refactoring, will come fix this after getting the 
# pendulum case to work (and we can actually visualize and verify the pendulum case)!

# -----Sample computation function-----------------------------------------------------------------
def compute_sample(args):
    ep, init_state, env = args
    init_state, init_states, init_controls, success = trainer.create_TO_init(ep, init_state)
    if not success: return None
    TO_states, TO_controls = trajopt.solve_iiwa_unconstrained_SQP(init_states, init_controls)
    RL_states, partial_rtg, next_states, done, rewards = trainer.compute_partial_rtg(
                                                            TO_controls, TO_states)
    return RL_states.tolist(), partial_rtg, next_states, done, sum(rewards)


if __name__ == '__main__':
    # -----Initialization--------------------------------------------------------------------------
    # set up conf
    PATH_TO_CONF = os.path.join(os.getcwd(), 'confs')
    sys.path.append(PATH_TO_CONF)
    conf = importlib.import_module('iiwa_conf')

    # initialze env, nn, buffer, TO, and rl
    env = getattr(conf, 'IiwaEnv')(conf)
    nn = ActorCriticNet(env, conf)
    buffer = ReplayBuffer(conf)
    trajopt = TrajOpt(env, conf)
    trainer = RLTrainer(env, nn, conf, 0)
    trainer.setup_model()

    # initialize episode reward arrays
    ep_arr_idx = 0
    ep_reward_arr = np.zeros(int(conf.TO_EPISODES*len(conf.NN_LOOPS))-ep_arr_idx)*np.nan
    update_step_counter = 0

    # -----Episode loop----------------------------------------------------------------------------
    for ep in range(len(conf.NN_LOOPS)):
        # collect samples
        print("Collecting samples...")
        init_rand_state = env.reset_batch(conf.TO_EPISODES)
        samples = [compute_sample((ep, init_rand_state[i, :], env)) for i in range(conf.TO_EPISODES)]
        valid_samples = [sample for sample in samples if sample]
        
        # add samples to replay buffer
        print(f"Compute_sample {len(valid_samples)}/{conf.TO_EPISODES} success")
        states, partial_rewards, state_nexts, dones, rewards = zip(*valid_samples)
        buffer.add(states, partial_rewards, state_nexts, dones)

        # Update nns and record rewards
        num_success = len(valid_samples)
        update_step_counter = trainer.learn_and_update(update_step_counter, buffer, ep)
        ep_reward_arr[ep_arr_idx : ep_arr_idx + num_success] = rewards
        for i in range(num_success):
            print("Episode  {}  --->   Return = {}".format(ep_arr_idx + i, rewards[i]))
        ep_arr_idx += num_success

        if update_step_counter > conf.NN_LOOPS_TOTAL:
            break
    
    trainer.RL_save_weights()

    

