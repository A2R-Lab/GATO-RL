import numpy as np
import pinocchio as pin
import sys
import importlib
import time
import os
import mujoco
import mujoco.viewer
from robot_descriptions.loaders.mujoco import load_robot_description

def test_iiwa_TO():
    # Add the src directory to the path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    from opt.traj_opt import TrajOpt
    from rl_trainer import RLTrainer

    PATH_TO_CONF = os.path.join(os.getcwd(), 'confs')
    sys.path.append(PATH_TO_CONF)
    conf = importlib.import_module('iiwa_conf')
    env = getattr(conf, 'IiwaEnv')(conf)

    TO_inst = TrajOpt(env, conf)
    T = conf.NSTEPS
    trainer = RLTrainer(env, TO_inst, conf, N_try=0)

    # Initialize and test the TO_solve method
    TO_inst = TrajOpt(env, conf)
    init_states = env.reset_batch(batch_size=10)
    init_traj_states = np.zeros((len(init_states), T + 1, conf.nx))
    success_count = 0
    for i in range(len(init_states)):
        init_traj_states, init_traj_controls, success = trainer.create_TO_init(0, init_states[i])
        if not success:
            print(f"Failed to create TO init for state {i}")
            continue
            
        traj_states, traj_controls = TO_inst.solve_iiwa_unconstrained_SQP(
            init_traj_states, init_traj_controls
        )
        cost = -env.reward(traj_states[-1])
        print("cost:", cost)
        print(env.ee(traj_states[0]))
        print(env.ee(traj_states[-1]))

        if cost < 1e-2:  # Assuming a cost threshold for success
            success_count += 1

    print(f"Total successful trajectories: {success_count}/{len(init_states)}")
    return traj_states, traj_controls


def viz_iiwa_traj(x: np.ndarray, _u: np.ndarray):
    """
    Display the trajectory using MuJoCo viewer.
    :param x: State trajectory (shape: [N, nx])
    :param _u: Control input trajectory (shape: [N-1, nu])
    """
    model = load_robot_description("iiwa14_mj_description")
    data = mujoco.MjData(model)
    
    nq = 7  # Number of joint positions
    nv = 7  # Number of joint velocities
    
    # Get trajectory dimensions
    N, nx = x.shape
    print(f"Visualizing trajectory with {N} time steps and {nx} state dimensions")
    
    # The state might include time as the last dimension, so extract only joint states
    if nx == 15:  # 7 positions + 7 velocities + 1 time = 15
        joint_states = x[:, :-1]  # Remove the last column (time)
        print("Detected time dimension in state, using first 14 dimensions for joint states")
    elif nx == 14:  # 7 positions + 7 velocities = 14
        joint_states = x
        print("Using all state dimensions for joint states")
    else:
        print(f"Warning: Unexpected state dimension {nx}, using first {2*nq} dimensions")
        joint_states = x[:, :2*nq]
    
    print(f"Joint states shape: {joint_states.shape}")
    
    while True:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for k in range(N):
                # Update joint positions and velocities
                data.qpos[:nq] = joint_states[k, :nq]
                data.qvel[:nv] = joint_states[k, nq:nq+nv]
                
                # Step the simulation
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.03)  # ~30 FPS, increase for slower animation
            
        print("Animation finished. Press Enter to replay or 'q' to quit.")
        user_input = input()
        if user_input.strip().lower() == 'q':
            break


if __name__ == "__main__":
    traj_states, traj_controls = test_iiwa_TO()   # your optimiser
    viz_iiwa_traj(traj_states, traj_controls)  # your visualiser
