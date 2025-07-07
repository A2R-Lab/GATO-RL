import numpy as np
import sys
import importlib
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from opt.traj_opt import TrajOpt
from rl_trainer import RLTrainer

# Load the double integrator config module
PATH_TO_CONF = os.path.join(os.getcwd(), 'confs')
sys.path.append(PATH_TO_CONF)
conf = importlib.import_module('double_int_conf')
env = getattr(conf, 'DoubleIntegratorEnv')(conf)
T = conf.NSTEPS
TO_inst = TrajOpt(env, conf)
trainer = RLTrainer(env, TO_inst, conf, N_try=0)

# Fill in initial states
init_states = env.reset_batch(batch_size=100)
init_traj_states = np.zeros((len(init_states), T + 1, conf.nx))
success_count = 0
for i in range(len(init_states)):
    init_traj_states, init_traj_controls, success = trainer.create_TO_init(0, init_states[i])
    traj_states, traj_controls, curr_iter, success = TO_inst.solve_double_integrator_unconstrained_SQP(
        init_traj_states, init_traj_controls, display_flag=False
    )
    np.set_printoptions(suppress=True, precision=4)
    flat = np.concatenate([traj_states.flatten(), traj_controls.flatten()])
    cost = env.running_cost(flat)
    print("cost:", cost)
    success_count += success

print(f"Total successful trajectories: {success_count}/{len(init_states)}")