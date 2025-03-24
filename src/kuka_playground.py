import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import sys
import importlib
import time
import os
from opt_control.traj_opt import TO

PATH_TO_CONF = os.path.join(os.getcwd(), 'confs')
sys.path.append(PATH_TO_CONF)

# Load the KUKA iiwa model (you can replace this with your own URDF)
URDF_PATH = os.path.join(os.getcwd(), 'urdfs/iiwa.urdf')
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
env = getattr(conf, 'Env')(conf)

TO_inst = TO(env, conf)

T = 10  # Number of time steps

ICS_state = np.random.rand(15)  # Random initial state (7 elements for KUKA IIWA)
init_TO_states = np.random.rand(T+1, 15)  # Random initial trajectory states
init_TO_controls = np.random.rand(T, 7)  # Random initial trajectory controls
print(ICS_state)
print(init_TO_states)
print(init_TO_controls)

# Initialize and test the TO_solve method
TO_inst = TO(env, conf)
TO_controls, TO_states, success_flag, TO_ee_pos_arr, TO_step_cost, _ = TO_inst.TO_solve(ICS_state, init_TO_states, init_TO_controls, T)

# Print results for inspection
print("Success Flag:", success_flag)
print("TO Controls:\n", TO_controls)
print("TO States:\n", TO_states)
print("TO End-Effector Positions:\n", TO_ee_pos_arr)
print("TO Step Costs:", TO_step_cost)
