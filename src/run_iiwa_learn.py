import sys
import os
import importlib
import numpy as np
from neural_network import ActorCriticNet
from replay_buffer import ReplayBuffer
from rl_trainer import RLTrainer
from opt.traj_opt import TrajOpt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# -----Sample computation function-----------------------------------------------------------------
def compute_sample(args):
    ep, init_state, env = args
    init_states, init_controls, success = trainer.create_TO_init(ep, init_state)
    if not success: return None
    TO_states, TO_controls = trajopt.solve_iiwa_unconstrained_SQP(init_states, init_controls)
    RL_states, partial_rtg, next_states, done, rewards = trainer.compute_partial_rtg(TO_controls, TO_states)
    if sum(rewards) < -1e3:  # Filter out poor samples
        print(f"Sample {ep} discarded due to low reward: {sum(rewards): .3f}")
        return None
    return RL_states, partial_rtg, next_states, done, sum(rewards)

def print_rewards(rewards, log_ptr):
    print(f"{'Episode':>8} | {'Rewards':>12}")
    print("-" * 30)
    for offset, reward in enumerate(rewards, start=log_ptr):
        print(f"{offset:8d} | {reward:12.3f}")
    print("-" * 30)
    print(f"{'Average':>8} | {np.mean(rewards):12.3f}")

if __name__ == '__main__':
    # -----Initialization--------------------------------------------------------------------------
    # Add conf path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'confs'))
    conf = importlib.import_module('iiwa_conf')

    # Initialize environment, neural network, replay buffer, trajectory optimizer, trainer
    env = getattr(conf, 'IiwaEnv')(conf)
    nn = ActorCriticNet(env, conf)
    buffer = ReplayBuffer(conf)
    trajopt = TrajOpt(env, conf)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    trainer = RLTrainer(env, nn, conf, timestamp)
    trainer.setup_model()

    TOTAL_EPISODES = conf.TO_EPISODES * len(conf.NN_LOOPS)
    reward_log = np.full(TOTAL_EPISODES, np.nan, dtype=float)
    log_ptr = 0
    update_step_ptr = 0

    # -----Episode loop----------------------------------------------------------------------------
    for ep in range(len(conf.NN_LOOPS)):
        print("Collecting samples...")
        init_rand_state = env.reset_batch(conf.TO_EPISODES)
        samples = [compute_sample((ep, init_rand_state[i, :], env)) for i in range(conf.TO_EPISODES)]
        samples = [s for s in samples if s]
        num_samples = len(samples)
        print(f"{num_samples}/{conf.TO_EPISODES} samples collected.")

        # Add samples to replay buffer
        states, partial_rewards, state_nexts, dones, rewards = zip(*samples)
        buffer.add(states, partial_rewards, state_nexts, dones)

        # Update neural networks and log rewards
        update_step_ptr = trainer.learn_and_update(update_step_ptr, buffer, ep)
        reward_log[log_ptr : log_ptr + num_samples] = rewards
        print_rewards(rewards, log_ptr)
        log_ptr += num_samples

        trainer.plot_training_curves()
        trainer.save_weights()
        trainer.save_conf()

        if update_step_ptr > conf.NN_LOOPS_TOTAL:
            break

    # Final saving and plotting
    trainer.plot_training_curves()
    trainer.save_weights()
    trainer.save_conf()