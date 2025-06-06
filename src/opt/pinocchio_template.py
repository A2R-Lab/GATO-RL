# copied from https://github.com/EmreAdabag/sqpcpu/blob/master/pinocchio_template.py

import pinocchio as pin
import numpy as np
np.set_printoptions(linewidth=99999999)

from scipy.sparse import bmat, csc_matrix, triu
import osqp


class thneed:

    def __init__(self, urdf_filename, N=32, dt=0.01, max_qp_iters=1, osqp_warm_start=True):
        self.stats = {
            'qp_iters': {
                'values': [],
                'unit': '',
                'multiplier': 1
            },
            'linesearch_alphas': {
                'values': [],
                'unit': '',
                'multiplier': 1
            },
            'sqp_stepsizes': {
                'values': [],
                'unit': '',
                'multiplier': 1
            }
        }
        # model things
        self.model = pin.buildModelFromUrdf(urdf_filename)
        self.data = self.model.createData()

        # environment
        self.N = N
        self.dt = dt

        # properties
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nx = self.nq + self.nv
        self.nu = len(self.model.joints) - 1 # honestly idk what this is about but it works for indy7
        self.nxu = self.nx + self.nu
        self.traj_len = (self.nx + self.nu)*self.N - self.nu
        
        # vars
        self.XU = np.zeros(self.traj_len)
        self.userho = False # EMRE make this reset
        self.rho = 1e-3
        self.drho = 1.0

        # cost
        self.dQ_cost = 0.01
        self.R_cost = 1e-5
        self.QN_cost = 100
        self.regularize = False
        self.eps = 1 # regularization parameter, velocities and controls *= (1 / (abs(norm(q) + eps)))

        
        # sparse matrix templates
        self.A = self.initialize_A()
        self.l = np.zeros(self.N*self.nx)
        self.P = self.initialize_P()
        self.g = np.zeros(self.traj_len)

        self.osqp = osqp.OSQP()
        osqp_settings = {'verbose': False, 'warm_start': osqp_warm_start}
        self.osqp.setup(P=self.P, q=self.g, A=self.A, l=self.l, u=self.l, **osqp_settings)

        # temporary variables so u don't have to pass around idk if this actually helps
        self.A_k = np.vstack([-1.0 * np.eye(self.nx), np.vstack([np.hstack([np.eye(self.nq), self.dt * np.eye(self.nq)]), np.ones((self.nq, 2*self.nq))])])
        self.B_k = np.vstack([np.zeros((self.nq, self.nq)), np.zeros((self.nq, self.nq))])
        self.cx_k = np.zeros(self.nx)
        # self.cv_k = np.zeros(self.nv)

        # hyperparameters
        self.max_qp_iters = max_qp_iters
        self.rho_factor = 1.2
        self.rho_min = 1e-3
        self.rho_max = 10
        self.mu = 10.0
    
        # random
        self.gravity = True

    def clean_start(self):
        self.XU = np.zeros(self.traj_len)

    def shift_start(self):
        self.XU[:-self.nxu] = self.XU[self.nxu:]
    
    def setxs(self, xs):
        self.XU[0:self.nx] = xs

    def gravity_off(self):
        self.model.gravity.linear = np.array([0,0,0])
    def gravity_on(self):
        self.model.gravity.linear = np.array([0,0,-9.81])

    def initialize_P(self):
        block = np.eye(self.nxu)
        block[:self.nq, :self.nq] = np.ones((self.nq, self.nq))
        bd = np.kron(np.eye(self.N), block)[:-self.nu, :-self.nu]
        return csc_matrix(triu(bd), shape=(self.traj_len, self.traj_len))
    
    def initialize_A(self):
        blocks = [[-1*np.ones((self.nx,self.nx))] + [None] * (2*self.N)]
        for i in range(self.N-1):
            row = []
            # Add initial zeros if needed
            for j in range(2*i):
                row.append(None)  # None is interpreted as zero block
            # Add A, B, I
            row.extend([np.ones((self.nx,self.nx)), 2 * np.ones((self.nx,self.nu)), -1 * np.ones((self.nx,self.nx))])
            # Pad remaining with zeros
            while len(row) < 2*self.N + 1:
                row.append(None)
            blocks.append(row)

        return bmat(blocks, format='csc')
    
    def compute_dynamics_jacobians(self, q, v, u):
        d_dq, d_dv, d_du = pin.computeABADerivatives(self.model, self.data, q, v, u)
        self.A_k[self.nx + self.nq:, :self.nq] = d_dq * self.dt
        self.A_k[self.nx + self.nq:, self.nq:2*self.nq] = d_dv * self.dt + np.eye(self.nv)
        self.B_k[self.nq:, :] = d_du * self.dt
        
        a = self.data.ddq
        qnext = pin.integrate(self.model, q, v * self.dt)
        vnext = v + a * self.dt
        xnext = np.hstack([qnext, vnext])
        xcur = np.hstack([q,v])
        self.cx_k = xnext - self.A_k[self.nx:] @ xcur - self.B_k @ u

    
    def update_constraint_matrix(self, xu, xs):
        # Fast update of the existing CSC matrix
        self.l[:self.nx] = -1 * xs # negative because top left is negative identity
        Aind = 0
        for k in range(self.N-1):
            xu_stride = (self.nx + self.nu)
            qcur = xu[k*xu_stride : k*xu_stride + self.nq]
            vcur = xu[k*xu_stride + self.nq : k*xu_stride + self.nx]
            ucur = xu[k*xu_stride + self.nx : (k+1)*xu_stride]
            
            self.compute_dynamics_jacobians(qcur, vcur, ucur)
            
            self.A.data[Aind:Aind+self.nx*self.nx*2]=self.A_k.T.reshape(-1)
            Aind += self.nx*self.nx*2
            self.A.data[Aind:Aind+self.nx*self.nu]=self.B_k.T.reshape(-1)
            Aind += self.nx*self.nu

            self.l[(k+1)*self.nx:(k+2)*self.nx] = -1.0 * self.cx_k

        self.A.data[Aind:] = -1.0 * np.eye(self.nx).reshape(-1)
        Aind += self.nx*self.nx

    def rk4(self,q,v,u,dt):    
        k1q = v
        k1v = pin.aba(self.model, self.data, q,v,u)
        q2 = pin.integrate(self.model, q, k1q * dt / 2)
        k2q = v + k1v * dt/2
        k2v = pin.aba(self.model, self.data, q2,k2q,u)
        q3 = pin.integrate(self.model, q, k2q * dt / 2)
        k3q = v + k2v * dt/2
        k3v = pin.aba(self.model, self.data, q3,k3q,u)
        q4 = pin.integrate(self.model, q, k3q * dt)
        k4q = v + k3v * dt
        k4v = pin.aba(self.model, self.data, q4,k4q,u)
        v_next = v + (dt/6) * (k1v + 2*k2v + 2*k3v + k4v)
        avg_v = (k1q + 2*k2q + 2*k3q + k4q) / 6
        q_next = pin.integrate(self.model, q, avg_v * dt)
        return q_next, v_next

    def eepos(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        return np.array(self.data.oMi[6].translation)

    def d_eepos(self, q):
        eepos_joint_id = 6
        pin.computeJointJacobians(self.model, self.data, q)
        deepos = pin.getJointJacobian(self.model, self.data, eepos_joint_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
        eepos = self.data.oMi[6].translation
        return eepos, deepos

    def update_cost_matrix(self, XU, eepos_g):
        Pind = 0
        for k in range(self.N):
            if k < self.N-1:
                XU_k = XU[k*(self.nx + self.nu) : (k+1)*(self.nx + self.nu)]
            else:
                XU_k = XU[k*(self.nx + self.nu) : (k+1)*(self.nx + self.nu)-self.nu]
            eepos, deepos = self.d_eepos(XU_k[:self.nq])
            eepos_err = np.array(eepos.T) - eepos_g[k*3:(k+1)*3]
            
            # cost multipliers
            dQ_modified = self.dQ_cost if not self.regularize else self.dQ_cost * (1/(abs(np.linalg.norm(eepos_err)) + self.eps))
            R_modified = self.R_cost if not self.regularize else self.R_cost * (1/(abs(np.linalg.norm(eepos_err)) + self.eps))
            Q_modified = self.QN_cost if k==self.N-1 else 1

            joint_err = eepos_err @ deepos

            g_start = k*(self.nx + self.nu)
            self.g[g_start : g_start + self.nx] = np.vstack([
                Q_modified * joint_err.T,
                (dQ_modified * XU_k[self.nq:self.nx]).reshape(-1)
            ]).reshape(-1)

            phessian = Q_modified * np.outer(joint_err, joint_err) + int(self.userho) * self.rho * np.eye(self.nq)
            pos_costs = phessian[np.tril_indices_from(phessian)]
            self.P.data[Pind:Pind+len(pos_costs)] = pos_costs
            Pind += len(pos_costs)
            self.P.data[Pind:Pind+self.nv] = np.full(self.nv, dQ_modified + int(self.userho) * self.rho)
            Pind+=self.nv
            if k < self.N-1:
                self.P.data[Pind:Pind+self.nu] = np.full(self.nu, R_modified + int(self.userho) * self.rho)
                Pind+=self.nu
                self.g[g_start + self.nx : g_start + self.nx + self.nu] = R_modified * XU_k[self.nx:self.nx+self.nu].reshape(-1)

    def setup_and_solve_qp(self, xu, xs, eepos_g):
        self.update_constraint_matrix(xu, xs)
        self.update_cost_matrix(xu, eepos_g)
        self.osqp.update(Px=self.P.data)
        self.osqp.update(Ax=self.A.data)
        self.osqp.update(q=self.g, l=self.l, u=self.l)
        return self.osqp.solve()
    
    def eepos_cost(self, eepos_goals, XU):
        qcost = 0
        vcost = 0
        ucost = 0
        dQ_modified = self.dQ_cost if not self.regularize else self.dQ_cost * (1/(abs(np.linalg.norm(eepos_err)) + self.eps))
        R_modified = self.R_cost if not self.regularize else self.R_cost * (1/(abs(np.linalg.norm(eepos_err)) + self.eps))
        for k in range(self.N):
            if k < self.N-1:
                XU_k = XU[k*(self.nx + self.nu) : (k+1)*(self.nx + self.nu)]
                Q_modified = 1
            else:
                XU_k = XU[k*(self.nx + self.nu) : (k+1)*(self.nx + self.nu)-self.nu]
                Q_modified = self.QN_cost
            eepos = self.eepos(XU_k[:self.nq])
            eepos_err = eepos.T - eepos_goals[k*3:(k+1)*3]
            qcost += Q_modified * np.linalg.norm(eepos_err)
            vcost += dQ_modified * np.linalg.norm(XU_k[self.nq:self.nx].reshape(-1))
            if k < self.N-1:
                ucost += R_modified * np.linalg.norm(XU_k[self.nx:self.nx+self.nu].reshape(-1))
        return qcost, vcost, ucost

    def integrator_err(self, XU):
        err = 0
        for k in range(self.N-1):
            xu_stride = (self.nx + self.nu)
            qcur = XU[k*xu_stride : k*xu_stride + self.nq]
            vcur = XU[k*xu_stride + self.nq : k*xu_stride + self.nx]
            ucur = XU[k*xu_stride + self.nx : (k+1)*xu_stride]

            a = pin.aba(self.model, self.data, qcur, vcur, ucur)
            qnext = pin.integrate(self.model, qcur, vcur*self.dt)
            vnext = vcur + a*self.dt

            qnext_err = qnext - XU[(k+1)*xu_stride : (k+1)*xu_stride + self.nq]
            vnext_err = vnext - XU[(k+1)*xu_stride + self.nq : (k+1)*xu_stride + self.nx]
            err += np.linalg.norm(qnext_err) + np.linalg.norm(vnext_err)
        return err

    def linesearch(self, XU, XU_fullstep, eepos_goals):
            base_qcost, base_vcost, base_ucost = self.eepos_cost(eepos_goals, XU)
            integrator_err = self.integrator_err(XU)
            baseCV = integrator_err + np.linalg.norm(XU[:self.nx] - XU[:self.nx])
            basemerit = base_qcost + base_vcost + base_ucost + self.mu * baseCV
            # print(f'base costs: {(base_qcost, base_vcost, base_ucost)}, baseCV: {baseCV}, base merit: {basemerit}')
            diff = XU_fullstep - XU

            alphas = np.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125])
            fail = True
            for alpha in alphas:
                XU_new = XU + alpha * diff
                qcost_new, vcost_new, ucost_new = self.eepos_cost(eepos_goals, XU_new)
                integrator_err = self.integrator_err(XU_new)
                CV_new = integrator_err + np.linalg.norm(XU_new[:self.nx] - XU[:self.nx])
                merit_new = qcost_new + vcost_new + ucost_new + self.mu * CV_new
                # print(f'new costs: {(qcost_new, vcost_new, ucost_new)}, newCV: {CV_new}, new merit: {merit_new}')
                exit_condition = (merit_new <= basemerit)

                if exit_condition:
                    fail = False
                    break

            if fail:
                alpha = 0.0
                self.drho = min(self.drho/self.rho_factor, 1 / self.rho_factor)
                self.rho = max(self.rho*self.drho, self.rho_min)
            
            self.stats['linesearch_alphas']['values'].append(alpha)
            return alpha
    
    def sqp(self, xcur, eepos_goals):
        updated = False
        for qp in range(self.max_qp_iters):

            sol = self.setup_and_solve_qp(self.XU, xcur, eepos_goals)
            
            alpha = self.linesearch(self.XU, sol.x, eepos_goals) if not np.any(sol.x==None) else 0.0
            # print(f'alpha: {alpha}')
            if alpha==0.0:
                continue
            updated = True
            step = alpha * (sol.x - self.XU)
            self.XU = self.XU + step
            
            stepsize = np.linalg.norm(step)
            self.stats['sqp_stepsizes']['values'].append(stepsize)

            if stepsize < 1e-3:
                break
        self.stats['qp_iters']['values'].append(qp+1)
        return not updated
    
# xpath = []
# def runeepos():
#     t = thneed(model)
#     nq = t.nq
#     nv = t.nv
#     nx = t.nx
#     nu = t.nu


#     xstart = np.hstack((np.ones(nq), np.zeros(nv)))
#     xcur = xstart
    
#     # endpoints = [thing.eepos(np.random.rand(6) * 2 - 1.0) for _ in range(3)]
#     endpoints = np.array([np.array(t.eepos(np.zeros(nq))), np.array(t.eepos(-0.8 * np.ones(nq)))])
#     print(endpoints)
#     endpoint_ind = 0
#     endpoint = endpoints[endpoint_ind]
#     eepos_goal = np.tile(endpoint, t.N).T

#     XU = np.zeros(t.N*(nx+t.nu)-t.nu)
#     XU = t.sqp(xcur, eepos_goal, XU)

#     print(f"costs: {t.eepos_cost(eepos_goal, XU)}")

#     for i in range(500):
        
#         # which goal are we planning to
#         cur_eepos = t.eepos(xcur[:nq])
#         goaldist = np.linalg.norm(cur_eepos - eepos_goal[:3])
#         if goaldist < 1e-1:
#             print('switching goals')
#             endpoint_ind = (endpoint_ind + 1) % len(endpoints)
#             endpoint = endpoints[endpoint_ind]
#             eepos_goal = np.tile(endpoint, t.N).T
#         print(goaldist)
#         if goaldist > 1.1:
#             print("breaking on big goal dist")
#             break

#         xu_new = t.sqp(xcur, eepos_goal, XU)
        
#         # print(f"costs: {t.eepos_cost(eepos_goal, XU)}")
#         trajopt_time = 0.01 # hard coded timestep
        
#         # simulate forward using old control
#         sim_time = trajopt_time
#         sim_steps = 0    # full steps taken
#         while sim_time > 0:
#             timestep = min(sim_time, t.dt)
#             # print(timestep)
#             control = XU[sim_steps*(nx+nu)+nx:(sim_steps+1)*(nx+nu)]
#             xcur = np.vstack(t.rk4(xcur[:nq], xcur[nq:nx], control, timestep)).reshape(-1)
            
#             if timestep > 0.5 * t.dt:
#                 sim_steps += 1
            
#             sim_time -= timestep
#             xpath.append(xcur[:nq])
#         if sim_steps > 0:
#             XU[:-(sim_steps)*(nx+nu) or len(XU)] = xu_new[(sim_steps)*(nx+nu):] # update XU with new traj
#         XU[:nx] = xcur.reshape(-1) # first state is current state
#         XU[-nx:] = np.hstack([np.ones(nq), np.zeros(nv)]) # last state is 0

#     return endpoints
# endpoints = runeepos()