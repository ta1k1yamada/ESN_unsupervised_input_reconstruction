import numpy as np
from numpy import inner, outer
from numpy.linalg import inv, pinv


def Fnorm(X):
    return np.sum( X**2 )

def RK(f, x0, ts):
    dt = ts[1]-ts[0]
    sol = np.zeros((len(ts), len(x0)))
    sol[0, :] = x0
    for it in range(len(ts)-1):
        x = sol[it, :]; t = ts[it]
        k1 = f(x, t)
        k2 = f(x + dt*k1/2, t + dt/2)
        k3 = f(x + dt*k2/2, t + dt/2)
        k4 = f(x + dt*k3, t + dt)
        sol[it+1, :] = x + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    return sol

def Lorenz63(x, y, z, sig, rho, beta):
    return np.array([sig*(y - x), x*(rho - z) - y, x*y - beta*z])



def calc_WD(D, R):
    return D@pinv(R)

def calc_Bhat(R, arcsig):
    return arcsig(R[:,1:])@pinv(R[:,:-1])

def calc_Bhat_reg(R, arcsig, nr, lam):
    return arcsig(R[:,1:])@R[:,:-1].T@pinv( R[:,:-1]@R[:,:-1].T + np.eye(nr)*lam)

def calc_WR(R, A, B, arcsig):
    Bh = calc_Bhat(R, arcsig)
    return pinv(A)@(Bh - B)

def RLS_step(x, y, A, P, lam):
    ilam = 1/lam
    v = y - A@x
    g = ilam*P@x / (1 + ilam*inner(x, P@x))
    P_new = ilam*(P - outer(g, P@x))
    A_new = A + outer(v, g)
    return P_new, A_new

def gen_ESN(nin, nr, Aamp, Brho, sigma):
    A = np.random.normal(0, Aamp, (nr, nin))
    B = np.random.normal(0, 1, (nr, nr))
    B *= Brho/np.max(np.abs(np.linalg.eigvals(B)))
    g = lambda d, r : sigma( A@d + B@r )
    return A, B, g

def EKF(rs_noise,F,G,H,Q,R):
    nr, niter = rs_noise.shape
    
    rs_filt_p = np.zeros((nr, niter))
    rs_filt_o = np.zeros((nr, niter))
    Ps_p = np.zeros((nr, nr, niter))
    Ps_p[:,:,0] = np.eye(nr)
    Ps_o = np.zeros((nr, nr, niter))
    Ps_o[:,:,0] = np.eye(nr)
    Ks = np.zeros((nr, nr, niter))

    for it in range(1,niter):
        rs_filt_p[:,it] = F@rs_filt_o[:,it-1] # prediction of current state
        Ps_p[:,:,it] = F@Ps_o[:,:,it-1]@F.T + G@Q@G.T
    
        Ks[:,:,it] = Ps_p[:,:,it]@H.T@inv(H@Ps_p[:,:,it]@H.T + R + np.eye(nr)*1e-5)
    
        rs_filt_o[:,it] = rs_filt_p[:,it] + Ks[:,:,it]@(rs_noise[:,it] - H@rs_filt_p[:,it]) # filter using observaiton
        Ps_o[:,:,it] = Ps_p[:,:,it] - Ks[:,:,it]@H@Ps_p[:,:,it]

    return rs_filt_p, rs_filt_o, Ps_p, Ps_o, Ks


def EKF_adapt_R(rs_noise,F,G,H,Q,alpha_R):
    nr, niter = rs_noise.shape
    
    rs_filt_p = np.zeros((nr, niter))
    rs_filt_o = np.zeros((nr, niter))
    Ps_p = np.zeros((nr, nr, niter))
    Ps_p[:,:,0] = np.eye(nr)
    Ps_o = np.zeros((nr, nr, niter))
    Ps_o[:,:,0] = np.eye(nr)
    Rs = np.zeros((nr, nr, niter))
    # Rs[:,:,0] = np.eye(nr)
    Rs[:,:,0] = Q
    Ks = np.zeros((nr, nr, niter))

    for it in range(1,niter):
        rs_filt_p[:,it] = F@rs_filt_o[:,it-1] # prediction of current state
        Ps_p[:,:,it] = F@Ps_o[:,:,it-1]@F.T + G@Q@G.T
    
        Ks[:,:,it] = Ps_p[:,:,it]@H.T@inv(H@Ps_p[:,:,it]@H.T + Rs[:,:,it-1] + np.eye(nr)*1e-5)
    
        rs_filt_o[:,it] = rs_filt_p[:,it] + Ks[:,:,it]@(rs_noise[:,it] - H@rs_filt_p[:,it]) # filter using observaiton
        Ps_o[:,:,it] = Ps_p[:,:,it] - Ks[:,:,it]@H@Ps_p[:,:,it]
    
        
        res = rs_noise[:, it] - H@rs_filt_o[:,it]
        R_est = outer(res,res) + H@Ps_o[:,:,it]@H.T
        # Rs[:,:,it] =  (1-alpha_R**it)*Rs[:,:,it-1] + alpha_R**it*R_est
        Rs[:,:,it] =  (1-alpha_R)*Rs[:,:,it-1] + alpha_R*R_est
    return rs_filt_p, rs_filt_o, Ps_p, Ps_o, Rs, Ks