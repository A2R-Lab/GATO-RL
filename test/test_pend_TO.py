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
env = getattr(conf, 'PendulumEnv')(conf, N_ts=T, u_min=20, u_max=20)

# Random initial state traj where each row is a state [theta (angle), w (angular vel)] at a time step
init_traj_states = np.random.rand(T+1, conf.nx + 1)
# Random initial trajectory controls
init_traj_controls = np.random.rand(T, conf.nu)

# Initialize and test the TO_solve method
TO_inst = TrajOpt(env, conf)
traj_states, traj_controls, curr_iter = TO_inst.solve_pend_constrained_SQP(init_traj_states,
                                                                           init_traj_controls,
                                                                           display_flag=True)
print("---------------------------------------------")
traj_states, traj_controls, curr_iter = TO_inst.solve_pend_unconstrained_SQP(init_traj_states,
                                                                             init_traj_controls,
                                                                             display_flag=True)