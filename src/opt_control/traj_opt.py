import numpy as np
import casadi as ca
from .pinocchio_template import thneed


class TO:
    def __init__(self, env, conf, w_S=0):
        self.env = env
        self.conf = conf

    def TO_Solve(self, ICS_state, init_TO_states, init_TO_controls):
        qpiters = 5
        dt = self.conf.dt
        N = init_TO_states.shape[0]
        pyt = thneed(self.conf.URDF_PATH, N=N, dt=dt, max_qp_iters=qpiters)

        for i in range(N):
            pyt.XU[i * (self.conf.nx + self.conf.na) : i * (self.conf.nx + self.conf.na) + self.conf.nx] = init_TO_states[i,:-1]
        for i in range(N-1):
            pyt.XU[i * (self.conf.nx + self.conf.na) + self.conf.nx : (i + 1) * (self.conf.nx + self.conf.na)] = init_TO_controls[i]

        xs = np.zeros(pyt.nx)  # Initial state
        eepos_g = 0.5 * np.ones(3 * pyt.N)
        pyt.setxs(xs)

        num_iters = 100
        # Run SQP optimization
        for i in range(num_iters):
            pyt.sqp(xs, eepos_g)
    
        X = np.array([pyt.XU[i * (self.conf.nx + self.conf.na) : i * (self.conf.nx + self.conf.na) + self.conf.nx] for i in range(N)]) 
        U = np.array([pyt.XU[i * (self.conf.nx + self.conf.na) + self.conf.nx : (i + 1) * (self.conf.nx + self.conf.na)] for i in range(N-1)])
        ee_pos_list = []
        for k in range(pyt.N):
            XU_k = pyt.XU[k * (self.conf.nx + self.conf.na) : k * (self.conf.nx + self.conf.na) + self.conf.nq]
            ee_pos = pyt.eepos(XU_k)
            ee_pos_list.append(ee_pos)
        ee_pos_arr = np.array(ee_pos_list)
        
        return X, U, ee_pos_arr