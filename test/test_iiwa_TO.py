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
init_TO_controls = np.random.rand(T, conf.na)    # Random initial trajectory controls (7 torques) (na=nu=number of controls)

print(ICS_state)
print(init_TO_states)
print(init_TO_controls)

# Initialize and test the TO_solve method
TO_inst = TrajOpt(env, conf)
traj_states, traj_controls = TO_inst.solve_iiwa_unconstrained_SQP(ICS_state, init_TO_states, init_TO_controls)

# Print results for inspection
print("traj controls:\n", traj_controls)
print("traj states:\n", traj_states)
