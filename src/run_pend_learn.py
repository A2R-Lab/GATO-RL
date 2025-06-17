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
    TO_states, TO_controls, iters = trajopt.solve_pend_constrained_SQP(init_states, init_controls)
    RL_states, partial_rtg, next_states, done, rewards = trainer.compute_partial_rtg(
                                                            TO_controls, TO_states)
    return RL_states.tolist(), partial_rtg, next_states, done, sum(rewards), iters

def print_rewards(rewards, iters, log_ptr):
    print(f"{'Episode':>8} | {'Rewards':>12} | {'SQP Iterations':>11}")
    print("-" * 36)

    for offset, (reward, iter) in enumerate(zip(rewards, iters), start=log_ptr):
        print(f"{offset:8d} | {reward:12.3f} | {iter:11d}")
    print("-" * 36)
    print(f"{'Average':>8} | {np.mean(rewards):12.3f} | { np.mean(iters):11.2f}")

if __name__ == '__main__':
    # -----Initialization--------------------------------------------------------------------------
    # set up conf
    PATH_TO_CONF = os.path.join(os.getcwd(), 'confs')
    sys.path.append(PATH_TO_CONF)
    conf = importlib.import_module('pendulum_conf')

    # initialze env, nn, buffer, TO, and rl
    env = getattr(conf, 'PendulumEnv')(conf, N_ts=300, u_min=5, u_max=5)
    nn = ActorCriticNet(env, conf)
    buffer = ReplayBuffer(conf)
    trajopt = TrajOpt(env, conf)
    trainer = RLTrainer(env, nn, conf, 1)
    trainer.setup_model()

    # initialize episode reward arrays
    TOTAL_EPISODES = conf.TO_EPISODES * len(conf.NN_LOOPS)
    reward_log = np.full(TOTAL_EPISODES, np.nan, dtype=float)
    log_ptr = 0
    update_step_ptr = 0

    # -----Episode loop----------------------------------------------------------------------------
    for ep in range(len(conf.NN_LOOPS)):
        # collect samples for TO_EPISODES episodes
        print("Collecting samples...")
        init_rand_state = env.reset_batch(conf.TO_EPISODES)
        samples = [compute_sample((ep, init_rand_state[i, :], env)) for i in range(conf.TO_EPISODES)]
        samples = [sample for sample in samples if sample]
        num_samples = len(samples)
        
        # add samples to replay buffer
        states, partial_rewards, state_nexts, dones, rewards, iters = zip(*samples)
        buffer.add(states, partial_rewards, state_nexts, dones)

        # Update nns and record rewards
        update_step_ptr = trainer.learn_and_update(update_step_ptr, buffer, ep)
        reward_log[log_ptr : log_ptr + num_samples] = rewards
        print_rewards(rewards, iters, log_ptr)
        log_ptr += num_samples

        if update_step_ptr > conf.NN_LOOPS_TOTAL:
            break
    
    trainer.RL_save_weights()

    
