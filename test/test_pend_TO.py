import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import sys
import importlib
import time
import os

# Add the src directory to the path
# NOTE: There should be a better way to handle this, but for now we will use this
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from opt.traj_opt import TrajOpt

PATH_TO_CONF = os.path.join(os.getcwd(), 'confs')
sys.path.append(PATH_TO_CONF)

T = 400  # Number of time steps

conf = importlib.import_module('pendulum_conf')
# Option 1: Pass additional keyword arguments
env = getattr(conf, 'PendulumEnv')(conf, N_ts=T, u_min=10, u_max=10)

# Random initial state traj where each row is a state [theta (angle), w (angular vel)] at a time step
# NOTE: We are appending a time step to the state, so the state dimension is conf.nx + 1. This is just used for the
# implementation of the algorithm and not a central part of the theory.
init_traj_states = np.random.rand(T+1, conf.nx+1)
# Random initial trajectory controls
init_traj_controls = np.random.rand(T, conf.nu)

# Reset to zero for testing initial conditions
#init_traj_states = np.zeros((T+1, conf.nx+1))

# Set the first state: theta = pi/2, w = 0.1 for init state testing
init_traj_states[0, 0] = np.pi/2  # theta
init_traj_states[0, 1] = -1      # w


# Initialize and test the TO_solve method
TO_inst = TrajOpt(env, conf)
traj_states, traj_controls, curr_iter, success = TO_inst.solve_pend_constrained_SQP(init_traj_states,
                                                                           init_traj_controls,
                                                                           display_flag=True)
print("---------------------------------------------")
traj_states, traj_controls, curr_iter, success = TO_inst.solve_pend_unconstrained_SQP(init_traj_states,
                                                                             init_traj_controls,
                                                                             display_flag=True)