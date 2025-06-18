import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import sys
import importlib
import time
import os

import mujoco
import mujoco.viewer
from robot_descriptions.loaders.mujoco import load_robot_description

def test_iiwa_TO():
    # Add the src directory to the path
    # NOTE: There should be a better way to handle this, but for now we will use this
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    from opt.traj_opt import TrajOpt

    PATH_TO_CONF = os.path.join(os.getcwd(), 'confs')
    sys.path.append(PATH_TO_CONF)

    # Load the KUKA iiwa model (you can replace this with your own URDF)
    URDF_PATH = os.path.join(os.getcwd(), 'confs/iiwa.urdf')
    robot = RobotWrapper.BuildFromURDF(URDF_PATH, [URDF_PATH])
    model = robot.model
    data = model.createData()

    # Get number of joints
    nq = model.nq  # Position state size
    nv = model.nv  # Velocity state size

    # Define initial state
    q = pin.neutral(model)  # Neutral joint positions
    v = np.zeros(nv)        # Zero initial velocity
    tau = np.random.uniform(-5, 5, nv)  # Random torque input

    # Print results
    print("Initial joint positions:", q)

    conf = importlib.import_module('iiwa_conf')
    env = getattr(conf, 'IiwaEnv')(conf)

    TO_inst = TrajOpt(env, conf)

    T = 100  # Number of time steps

    # KUKA IIWA has 7 joints, so:
    # - State dimension: 14 (7 positions + 7 velocities) 
    # - Control dimension: 7 (one torque per joint)
    ICS_state = np.random.rand(14)  # Random initial state (14 elements: 7 positions + 7 velocities)
    init_TO_states = np.random.rand(T+1, conf.nx+1)  # Random initial trajectory states (14 state + 1 timestep)
    init_TO_controls = np.random.rand(T, conf.nu)    # Random initial trajectory controls (7 torques) (na=nu=number of controls)

    print(init_TO_controls.shape)
    print(init_TO_states.shape)

    # Initialize and test the TO_solve method
    TO_inst = TrajOpt(env, conf)
    traj_states, traj_controls = TO_inst.solve_iiwa_unconstrained_SQP(init_TO_states, init_TO_controls)

    # Print results for inspection
    print("traj controls:\n", traj_controls.shape)
    print("traj states:\n", traj_states.shape)

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
