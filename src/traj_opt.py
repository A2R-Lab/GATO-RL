import numpy as np
import casadi as ca

class TO:
    def __init__(self, env, conf, w_S=0):
        self.env = env
        self.conf = conf
        cx = ca.MX.sym("x",self.conf.nx,1)
        cu = ca.MX.sym("u",self.conf.na,1)

        self.x_next = ca.Function('x_next', [cx, cu], [self.env.simulate(cx,cu)])
        self.cost = ca.Function('cost', [cx,cu], [self.env.cost(cx,cu)])
        self.p_ee = ca.Function('p_ee', [cx], [self.env.ee(cx)])
    
    def TO_cpu_solve(self, ICS_state, init_TO_states, init_TO_controls, T):
        opti = ca.Opti()
        total_cost = 0
        step_costs = []
        xs = [opti.variable(self.conf.nx) for _ in range(T+1)]
        us = [opti.variable(self.conf.na) for _ in range(T)]

        # define cost function
        opti.subject_to(xs[0] == ICS_state[:-1])

        for t in range(T):
            x_next = self.x_next(xs[t], us[t])
            opti.subject_to(xs[t + 1] == x_next)
            r_cost = self.cost(xs[t], us[t])
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
        except:
            TO_states = np.array([ opti.debug.value(x) for x in xs ])
            TO_controls = np.array([ opti.debug.value(u) for u in us ])
            TO_total_cost, TO_ee_pos_arr, TO_step_cost = None, None, None
            success_flag = 0

        return success_flag, TO_controls, TO_states, TO_ee_pos_arr, TO_total_cost, TO_step_cost
        
        

    def TO_solve(self, ICS_state, init_TO_states, init_TO_controls, T):
        success_flag, TO_controls, TO_states, TO_ee_pos_arr, _, TO_step_cost = self.TO_cpu_solve(ICS_state, init_TO_states, init_TO_controls, T)
        if success_flag == 0:
            return None, None, success_flag, None, None, None 

        # Add the last state component (time)
        TO_states = np.concatenate((TO_states, init_TO_states[0,-1] + np.transpose(self.conf.dt*np.array([range(T+1)]))), axis=1)
            
        return TO_controls, TO_states, success_flag, TO_ee_pos_arr, TO_step_cost
