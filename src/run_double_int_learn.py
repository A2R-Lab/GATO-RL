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
import torch

# NOTE: Currently broken from refactoring, will come fix this after getting the 
# pendulum case to work (and we can actually visualize and verify the pendulum case)!

# -----Sample computation function-----------------------------------------------------------------
def compute_sample(args):
    ep, init_state, env = args
    init_states, init_controls, success = trainer.create_TO_init(ep, init_state)
    if not success: return None
    TO_states, TO_controls, iters, success = trajopt.solve_double_integrator_unconstrained_SQP(init_states, init_controls)
    # if not success: return None
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

def plot_rtg_and_critic(env, conf, ac_net, trainer):
    actor, critic = trainer.actor_model, trainer.critic_model
    x = np.linspace(-1, 1, 50)
    v = np.linspace(-1, 1, 50)
    X, V = np.meshgrid(x, v)
    T = np.zeros_like(X)

    # rtg
    rtg_values = np.zeros_like(X)
    for idx, (p, v) in enumerate(zip(X.ravel(), V.ravel())):
        init_state = np.array([p, v, 0.0])
        states = np.zeros((conf.NSTEPS + 1, conf.nx + 1))
        states[0] = init_state
        actions = np.zeros((conf.NSTEPS, conf.nu))
        for t in range(conf.NSTEPS):
            state_tensor = torch.tensor(states[t][None], dtype=torch.float32)
            with torch.no_grad():
                action = ac_net.eval(actor, state_tensor, is_actor=True).cpu().numpy().squeeze()
            actions[t] = action
            states[t + 1] = env.simulate(states[t], actions[t])
        _, rtg, _, _, _ = trainer.compute_partial_rtg(actions, states)
        rtg_values.ravel()[idx] = rtg[0]

    # critic values
    states = np.stack([X.ravel(), V.ravel(), T.ravel()], axis=1)
    states_tensor = torch.tensor(states, dtype=torch.float32)
    with torch.no_grad():
        values = ac_net.eval(critic, states_tensor).cpu().numpy().reshape(X.shape)

    # plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    c1 = axs[0].contourf(X, V, rtg_values, levels=30, cmap='plasma')
    fig.colorbar(c1, ax=axs[0], label='Reward-to-Go')
    axs[0].set_title('Reward-to-Go (Actor Policy)')
    axs[0].set_xlabel('Position')
    axs[0].set_ylabel('Velocity')

    c2 = axs[1].contourf(X, V, values, levels=30, cmap='plasma')
    fig.colorbar(c2, ax=axs[1], label='Critic Value')
    axs[1].set_title('Critic Value Function')
    axs[1].set_xlabel('Position')
    axs[1].set_ylabel('Velocity')
    plt.tight_layout()
    plt.savefig(f"{trainer.path}/critic_vals.png", dpi=300)
    print(f"Critic values saved to {trainer.path}/critic_vals.png.")

def rollout_actor_trajectory(env, conf, ac_net, trainer):
    actor = trainer.actor_model
    T = 0.0
    init_state = np.array([1.0, 1.0, T])
    steps = conf.NSTEPS - int(T / conf.dt)
    states = np.zeros((steps + 1, conf.nx + 1))
    actions = np.zeros((steps, conf.nu))
    states[0] = init_state

    log_path = os.path.join(trainer.path, "actor_traj.log")
    with open(log_path, "w") as f:
        f.write(f"{'t':>4} | {'x':>8} | {'v':>8} | {'u':>8}\n")
        f.write("-" * 40 + "\n")
        for t in range(steps):
            state_tensor = torch.tensor(states[t][None], dtype=torch.float32)
            with torch.no_grad():
                action = ac_net.eval(actor, state_tensor, is_actor=True).cpu().numpy().reshape(-1)
            actions[t] = action
            f.write(f"{t:4d} | {states[t, 0]:8.3f} | {states[t, 1]:8.3f} | {action.item():8.3f}\n")
            states[t + 1] = env.simulate(states[t], action)

    # Plot results
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.plot(states[:, 0], label='Position')
    plt.plot(states[:, 1], label='Velocity')
    plt.xlabel('Time step')
    plt.title('State Trajectory')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(actions, label='Action')
    plt.xlabel('Time step')
    plt.title('Actor Actions')
    plt.legend()

    plt.tight_layout()
    fig_path = os.path.join(trainer.path, "actor_traj.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Actor trajectory logged to {log_path} and plotted to {fig_path}.")

if __name__ == '__main__':
    # -----Initialization--------------------------------------------------------------------------
    # set up conf
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'confs'))
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
    plot_rtg_and_critic(env, conf, nn, trainer)
    rollout_actor_trajectory(env, conf, nn, trainer)
    log_file.close()


    
