import numpy as np
import casadi as ca
from .pinocchio_template import thneed


class TO:
    def __init__(self, env, conf, w_S=0):
        self.env = env
        self.conf = conf
        
    def TO_cpu_solve(self, ICS_state, init_TO_states, init_TO_controls, T):
        opti = ca.Opti()

        total_cost = 0
        step_costs = []
        xs = [opti.variable(self.conf.nx) for _ in range(T+1)]
        us = [opti.variable(self.conf.na) for _ in range(T)]

        # define cost function
        opti.subject_to(xs[0] == ICS_state[:-1])

        for t in range(T):
            state_next = self.env.x_next(xs[t], us[t])
            opti.subject_to(xs[t + 1] == state_next)
            r_cost = self.env.cost(xs[t],us[t])
            total_cost += r_cost
            step_costs.append(r_cost)

        opti.minimize(total_cost) 
        # set initial guesses of TO states and controls
        init_x_TO = [np.array(init_TO_states[i,:-1]) for i in range(T+1)]
        init_u_TO = [np.array(init_TO_controls[i,:]) for i in range(T)]
        for x,xg in zip(xs,init_x_TO): opti.set_initial(x,xg)
        for u,ug in zip(us,init_u_TO): opti.set_initial(u,ug)

        opts = {'ipopt.sb': 'yes','ipopt.print_level': 0, 'print_time': 0}
        opti.solver("ipopt", opts) 
        try:
            opti.solve()
            TO_states = np.array([ opti.value(x) for x in xs ])
            TO_controls = np.array([ opti.value(u) for u in us ])
            TO_total_cost = opti.value(total_cost)
            TO_ee_pos_arr = np.array([self.p_ee(TO_states[n, :]).full().flatten() for n in range(T+1)])
            TO_step_cost = np.array([opti.value(c) for c in step_costs])
            success_flag = 1
        except Exception as e:
            print(f"Trajopt failed because of exception: {e}")
            TO_states = np.array([ opti.debug.value(x) for x in xs ])
            TO_controls = np.array([ opti.debug.value(u) for u in us ])
            TO_total_cost, TO_ee_pos_arr, TO_step_cost = None, None, None
            success_flag = 0

        return success_flag, TO_controls, TO_states, TO_ee_pos_arr, TO_total_cost, TO_step_cost
        

    def TO_Solve(self, ICS_state, init_TO_states, init_TO_controls, T):
        qpiters = 5
        N = self.conf.NSTEPS
        dt = self.conf.dt
        pyt = thneed("urdfs/iiwa.urdf", N=N, dt=dt, max_qp_iters=qpiters)
        xs = np.zeros(pyt.nx)  # Initial state
        eepos_g = 0.5 * np.ones(3 * pyt.N)   # End-effector position goals
        pyt.setxs(xs)

        num_iters = 100
        # Run SQP optimization
        for i in range(num_iters):
            pyt.sqp(xs, eepos_g)
    
        X = np.array([pyt.XU[i * (self.conf.nx + self.conf.na) : i * (self.conf.nx + self.conf.na) + self.conf.nx] for i in range(N)]) 
        U = np.array([pyt.XU[i * (self.conf.nx + self.conf.na) + self.conf.nx : (i + 1) * (self.conf.nx + self.conf.na)] for i in range(N-1)])
        print(pyt.XU.shape, X.shape, U.shape)
        ee_pos_list = []
        for k in range(pyt.N):
            XU_k = pyt.XU[k * (self.conf.nx + self.conf.na) : k * (self.conf.nx + self.conf.na) + self.conf.nq]
            ee_pos = pyt.eepos(XU_k)
            ee_pos_list.append(ee_pos)
        ee_pos_arr = np.array(ee_pos_list)
        
        return X, U, ee_pos_arr