import numpy as np
import sys
import importlib
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from opt.traj_opt import TrajOpt  # Assumes your SQP class is in opt/traj_opt.py

# Load the double integrator config module
PATH_TO_CONF = os.path.join(os.getcwd(), 'confs')
sys.path.append(PATH_TO_CONF)

T = 300  # Number of time steps

conf = importlib.import_module('double_int_conf')
env = getattr(conf, 'DoubleIntegratorEnv')(conf)

# Create random initial state trajectory: shape (T+1, conf.nx + 1)
init_traj_states = np.zeros((T + 1, conf.nx + 1))  # [x, v, t]
init_traj_controls = np.zeros((T, conf.nu))       # [u]

# Set first state to a specific condition (x=1.0, v=-1.0)
init_traj_states[0, 0] = -1.0   # x
init_traj_states[0, 1] = 0.0  # v

# Initialize the Trajectory Optimizer
TO_inst = TrajOpt(env, conf)

# Solve unconstrained SQP
traj_states, traj_controls, curr_iter, success = TO_inst.solve_double_integrator_unconstrained_SQP(
    init_traj_states, init_traj_controls, display_flag=True
)
np.set_printoptions(suppress=True, precision=4)
print("states:", traj_states)
print("controls:", traj_controls)
print("success", success)