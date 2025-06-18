import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import sys
import importlib
import time
import os

import mujoco
from mujoco import viewer
import mujoco_viewer # Trying out this 3rd party viewer

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

    T = 10  # Number of time steps

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


def visualize_iiwa_trajectory(traj_states,
                              urdf_path="confs/iiwa.urdf",
                              fps=30):
    """
    Display a joint-space trajectory for a 7-DOF KUKA IIWA.

    traj_states : (T+1, 14) ndarray
        q = traj_states[:, :7]   (rad)
        qd = traj_states[:, 7:]  (optional, rad/s)
    urdf_path   : str
    fps         : int
    """
    # MuJoCo can compile MJCF **or URDF** files directly
    model = mujoco.MjModel.from_xml_path(urdf_path)     # ← no mjcf module
    data  = mujoco.MjData(model)

    dt_frame   = 1.0 / fps
    n_frames   = traj_states.shape[0]
    next_frame = 0
    last_tick  = time.time()

    with viewer.launch_passive(model, data) as v:
        print("Viewer open — close the window or press ESC to quit.")
        while v.is_running():
            now = time.time()
            if now - last_tick >= dt_frame and next_frame < n_frames:
                q = traj_states[next_frame, :7]
                data.qpos[:7] = q
                if traj_states.shape[1] >= 14:
                    data.qvel[:7] = traj_states[next_frame, 7:14]

                mujoco.mj_forward(model, data)
                next_frame += 1
                last_tick = now
            v.sync()

def visualize_iiwa_trajectory2(traj_states,
                              urdf_path="confs/iiwa.urdf",
                              fps=30):
    """
    Display a joint-space trajectory for a 7-DOF KUKA IIWA.
    """
    try:
        # Try to load URDF directly
        model = mujoco.MjModel.from_xml_path(urdf_path)
    except Exception as e:
        print(f"Error loading URDF: {e}")
        print("Try converting URDF to MJCF format first")
        return
    
    data = mujoco.MjData(model)
    
    # Debug: Print model info
    print(f"Model loaded successfully!")
    print(f"Number of DOFs: {model.nq}")
    print(f"Number of bodies: {model.nbody}")
    print(f"Trajectory shape: {traj_states.shape}")
    
    # Set initial pose to see if robot appears
    if model.nq >= 7:
        data.qpos[:7] = traj_states[0, :7]
        mujoco.mj_forward(model, data)
    
    dt_frame = 1.0 / fps
    n_frames = traj_states.shape[0]
    next_frame = 0
    last_tick = time.time()


    """# Trying out use of the 3rd party viewer
    mj_viewer = mujoco_viewer.MujocoViewer(model, data)
    # simulate and render
    for _ in range(10000):
        if mj_viewer.is_alive:
            mujoco.mj_step(model, data)
        else:
            break
    mj_viewer.close()
    return 0"""

    with viewer.launch_passive(model, data) as v:
        print("Viewer open — close the window or press ESC to quit.")
        
        # Force initial update
        v.sync()
        
        while v.is_running():
            now = time.time()
            if now - last_tick >= dt_frame and next_frame < n_frames:
                # Handle the 15-dimensional state (adjust indexing as needed)
                q = traj_states[next_frame, :7]  # First 7 elements as joint positions
                
                if model.nq >= 7:
                    data.qpos[:7] = q
                    if traj_states.shape[1] >= 14:
                        # Adjust this based on your actual state structure
                        data.qvel[:7] = traj_states[next_frame, 7:14]
                
                mujoco.mj_forward(model, data)
                next_frame += 1
                last_tick = now
                
                # Reset animation when finished
                if next_frame >= n_frames:
                    next_frame = 0
            
            v.sync()


if __name__ == "__main__":
    traj_states, traj_controls = test_iiwa_TO()   # your optimiser
    visualize_iiwa_trajectory2(traj_states,
                              urdf_path=os.path.join(os.getcwd(),
                                                     "confs/iiwa.urdf"))
