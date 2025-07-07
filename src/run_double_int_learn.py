import sys
import os
import importlib
import numpy as np
from neural_network import ActorCriticNet
from replay_buffer import ReplayBuffer
from rl_trainer import RLTrainer
from opt.traj_opt import TrajOpt
import matplotlib.pyplot as plt
from datetime import datetime

# NOTE: Currently broken from refactoring, will come fix this after getting the 
# pendulum case to work (and we can actually visualize and verify the pendulum case)!

# -----Sample computation function-----------------------------------------------------------------
def compute_sample(args):
    ep, init_state, env = args
    init_states, init_controls, success = trainer.create_TO_init(ep, init_state)
    if not success: return None
    TO_states, TO_controls, iters, success = trajopt.solve_double_integrator_unconstrained_SQP(init_states, init_controls)
    if not success: return None
    RL_states, partial_rtg, next_states, done, rewards = trainer.compute_partial_rtg(
                                                            TO_controls, TO_states)
    return RL_states, partial_rtg, next_states, done, sum(rewards), iters

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
    conf = importlib.import_module('double_int_conf')

    # initialze env, nn, buffer, TO, and rl
    env = getattr(conf, 'DoubleIntegratorEnv')(conf)
    nn = ActorCriticNet(env, conf)
    buffer = ReplayBuffer(conf)
    trajopt = TrajOpt(env, conf)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    trainer = RLTrainer(env, nn, conf, timestamp)
    trainer.setup_model()

    # initialize episode reward arrays
    TOTAL_EPISODES = conf.TO_EPISODES * len(conf.NN_LOOPS)
    reward_log = np.full(TOTAL_EPISODES, np.nan, dtype=float)
    log_ptr = 0
    update_step_ptr = 0

    log_file_path = f"double_int/{timestamp}/buffer_log.txt"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    log_file = open(log_file_path, "w")

    # -----Episode loop----------------------------------------------------------------------------
    for ep in range(len(conf.NN_LOOPS)):
        # collect samples for TO_EPISODES episodes
        print("Collecting samples...")
        init_rand_state = env.reset_batch(conf.TO_EPISODES)
        samples = [compute_sample((ep, init_rand_state[i, :], env)) for i in range(conf.TO_EPISODES)]
        samples = [sample for sample in samples if sample]
        num_samples = len(samples)
        print(f"{num_samples}/{conf.TO_EPISODES} samples collected.")
        
        # add samples to replay buffer
        states, partial_rewards, state_nexts, dones, rewards, iters = zip(*samples)
        buffer.add(states, partial_rewards, state_nexts, dones)
        np.set_printoptions(suppress=True)
        for i in range(num_samples):
            log_file.write(f"Episode {log_ptr + i}:\n")
            log_file.write(f"States:\n{np.array(states[i])}\n")
            log_file.write(f"Partial Rewards:\n{np.array(partial_rewards[i])}\n")
            log_file.write(f"Next States:\n{np.array(state_nexts[i])}\n")
            log_file.write(f"Dones:\n{np.array(dones[i])}\n")
            log_file.write(f"Total Reward: {rewards[i]}\n")
            log_file.write(f"SQP Iterations: {iters[i]}\n")
            log_file.write("-" * 40 + "\n")

        # Update nns and record rewards
        update_step_ptr = trainer.learn_and_update(update_step_ptr, buffer, ep)
        reward_log[log_ptr : log_ptr + num_samples] = rewards
        print_rewards(rewards, iters, log_ptr)
        log_ptr += num_samples
        trainer.plot_training_curves()
        trainer.save_weights()
        trainer.save_conf()

        if update_step_ptr > conf.NN_LOOPS_TOTAL:
            break

    trainer.plot_training_curves()
    trainer.save_weights()
    trainer.save_conf()

    
