"""
	CYAxiverse.jlm_python
Here we wrap Joan's optimization code to use -- to be rewritten in `julia` when time allows
"""
module jlm_python_test
using PyCall
using ..structs: Solver1D, SolverND, Min_JLM_ND, Min_JLM_1D
function __init__()
    py"""
    import numpy as np
    from numpy.linalg import matrix_rank
    import random
    from scipy.integrate import solve_ivp
    from scipy import optimize

    ############################
    ## single axion solver######
    ############################
    def ode(t,theta,L_log,L_sign,alphas,phases):
        return 1

    def zero_V_prime(t,theta,L_log,L_sign,alphas,phases):
        vec_sin = alphas*np.sin(theta*alphas.T + np.append([0],phases)).T
        lambdas = L_sign * 10.**(L_log - L_log[0])
        V_prime = np.dot(lambdas,vec_sin)
        return V_prime

    def potential_V(theta,lambdas,alphas,phases):
        V = 0
        phase_vec = np.append([0],phases)
        for i in range(len(alphas)):
            V = V + lambdas[i]*(1-np.cos(theta*alphas[i] + phase_vec[i]))
        return V

    def potential_V_prime(theta,lambdas,alphas,phases):
        V_prime = 0
        phase_vec = np.append([0],phases)
        for i in range(len(alphas)):
            V_prime = V_prime + lambdas[i]*alphas[i]*(np.sin(theta*alphas[i] + phase_vec[i]))
        return V_prime

    def hessian_V(theta,L_log,L_sign,alphas,phases):
        V__double_prime = 0
        phase_vec = np.append([0],phases)
        lambdas = L_sign * 10.**(L_log - L_log[0])
        for i in range(len(alphas)):
            V__double_prime = V__double_prime + lambdas[i]*(alphas[i]**2)*(np.cos(theta*alphas[i] + phase_vec[i]))
        return V__double_prime

    def one_dim_axion_solver(domain,Q,L_log,L_sign,det_QTilde,phases,Z,inv_symmetries,det_Sym):
        sol_v = solve_ivp(ode, (0,domain),(0,), max_step = 0.01, events=[zero_V_prime], args =[L_log,L_sign,Q,phases])
        points_x = sol_v.t
        points_y = sol_v.y
        extrema_1D = sol_v.t_events[0] #all the extrema of the 1D potential
        n_extrema_1D =  extrema_1D.shape[0]

        #### AMONG ALL THE EXTREMA WE GET WHICH ARE MINIMA OF THE POTENTIAL ####
        min_list_1D = []
        for v in range(n_extrema_1D):
            if np.sign(hessian_V(extrema_1D[v],L_log,L_sign,Q,phases)) > 0:
                min_list_1D = np.append(min_list_1D,v)

        
        theta_min_1D = extrema_1D[min_list_1D.astype(int)]

        Thetas_total = np.around(theta_min_1D,6)

        ###### 	ALGORITHM TO GET UNIQUE MINIMA ######
        if Z == 1: #ALL alpha-vectors have integer entries so we mod by 2pi
            Thetas_total_mod = Thetas_total % (2*np.pi) #position of minima between 0 and 2pi this helps to distinguish them!
            theta_mask = np.where(np.abs(Thetas_total_mod) > 1e-04, Thetas_total_mod, 0)
            theta_mask_2pi = np.where(np.abs(np.abs(theta_mask)-2*np.pi) > 1e-04, theta_mask, 0)
            theta_mask_pi = np.where(np.abs(np.abs(theta_mask_2pi) - np.pi) > 1e-04, theta_mask_2pi, np.pi)

            ###### WE FIND THE UNIQUE VECTORS IN THE LIST ######
            Theta_unique_mod = np.unique(theta_mask_pi,axis=0)
            Theta_round = np.where(Theta_unique_mod==np.pi , Theta_unique_mod, np.around(Theta_unique_mod, decimals=4)) #decimals=6
            Theta_unique_final = np.unique(Theta_round, axis=0)

            N_unique = Theta_unique_final.shape[0]
            Total_min_theory = np.abs(det_QTilde * N_unique)
        else:	#at least one entry has a rational number
            c_vector = inv_symmetries * Thetas_total / (2*np.pi)
            c_vector_mod = c_vector.copy()
            c_vector_mod = c_vector_mod % 1
            
            ###### WE FIND THE UNIQUE VECTORS IN THE LIST ######
            c_vector_mask = np.where(np.abs(c_vector_mod) > 1e-04, c_vector_mod, 0)
            c_vector_mask_1 = np.where(np.abs(np.abs(c_vector_mask)- 1) > 1e-04, c_vector_mask, 0)

            ###### WE FIND THE UNIQUE VECTORS IN THE LIST ######
            c_vector_unique_mod = np.unique(c_vector_mask_1,axis=0)
            c_vector_round = np.around(c_vector_unique_mod, decimals=4)
            c_vector_unique_final = np.unique(c_vector_round, axis=0)

            #WE CONSRTUCT THE VECTOR OF THETA-MIN
            symmetries = 1 / inv_symmetries
            Theta_unique_final = symmetries * c_vector_unique_final * (2*np.pi)

            #TOTAL MIN FOR RATIONAL ALPHA-VECTORS
            N_unique = Theta_unique_final.shape[0]
            Total_min_theory = np.abs((det_QTilde/det_Sym) * N_unique)

        return Total_min_theory, Theta_unique_final, det_QTilde


    ############################
    ## Multiaxion solver #######
    ############################
    def Nabla(x,Q,L_log,L_sign,phases):
        N = np.shape(Q)[1] #number of axions
        arg_sin = np.matmul(Q[N:,:],x) #argument of perturbing term
    
        Ratio_lambda = 10.**(np.subtract.outer(L_log[N:], L_log[0:N].T))
        Ratio_lambda = np.where(np.isinf(Ratio_lambda), 0, Ratio_lambda)
        Ratio_lambda = np.where(np.isnan(Ratio_lambda), 0, Ratio_lambda)
    
        Ratio_Q_eff = Ratio_lambda * Q[N:,:]
        Ratio_Q_eff = np.matmul(np.diag(L_sign[N:]),Ratio_Q_eff)
    
        sin_pert = np.sin(arg_sin + phases)
        perturbation = np.matmul(Ratio_Q_eff.T,sin_pert)
    
        #Grad_i ARE FUNCTIONS WE NEED TO FIND ROOTS OF
        Grad_i = np.sign(L_sign[0:N])*np.sin(x) + perturbation
    
        return Grad_i
        
    def Hess(x,Q,L_log,L_sign,phases):
        N = np.shape(Q)[1]
        arg_cos = np.matmul(Q[N:,:],x)
    
        Ratio_lambda = 10.**(np.subtract.outer(L_log[N:], L_log[0:N].T))
        Ratio_lambda = np.where(np.isinf(Ratio_lambda), 0, Ratio_lambda)
        Ratio_lambda = np.where(np.isnan(Ratio_lambda), 0, Ratio_lambda)
    
        Ratio_Q_eff = Ratio_lambda * Q[N:,:]
        Ratio_Q_eff = np.matmul(np.diag(L_sign[N:]),Ratio_Q_eff)
    
        cos_pert = np.cos(arg_cos + phases)
        cos_matrix = np.matmul(np.diag(cos_pert),Q[N:,:])
    
        #PERTURBATION TO HESSIAN: h_ij
        h_ij = np.matmul(Ratio_Q_eff.T,cos_matrix)
    
        #H_ij = GRADIENT OF "Grad_i" = GRADIENT OF EACH FUNCTION WE WANT TO FIND ROOTS OF
        H_ij = np.diag(np.sign(L_sign[0:N])*np.cos(x)) + h_ij #Hessian matrix
    
        return H_ij
        
    def Hessian_true(x,Q,L_log,L_sign,phases):
        N = np.shape(Q)[1]
        arg_cos = np.matmul(Q[N:,:],x)
    
        Ratio_lambda = 10.**(np.subtract.outer(L_log[N:], L_log[0:N].T)/2)
        Ratio_lambda = np.where(np.isinf(Ratio_lambda), 0, Ratio_lambda)
        Ratio_lambda = np.where(np.isnan(Ratio_lambda), 0, Ratio_lambda)
        Ratio_lambda = Ratio_lambda * Q[N:,:]
    
        Ratio_lambda_sign = np.matmul(np.diag(L_sign[N:]),Ratio_lambda)
        cos_pert = np.cos(arg_cos + phases)
        cos_matrix = np.matmul(np.diag(cos_pert),Ratio_lambda_sign)
    
        #PERTURBATION TO HESSIAN: h_ij
        h_ij = np.matmul(Ratio_lambda.T,cos_matrix)
    
        #HESSIAN IN NEW COORDINATES. WE WILL FIND EIGENVALUES OF H_ij TO DETERMINE WHAT EXTREMA ARE MINIMA
        H_ij = np.diag(L_sign[0:N]*np.cos(x)) + h_ij #Hessian matrix
        return H_ij
    
    def multi_axion_solver(samples,Q,L_log,L_sign,det_QTilde,phases,Z,inv_symmetries,det_Sym):
        N_axions = np.shape(Q)[1]
        if Z == 1:
            theta_0 = np.random.uniform(0,2*np.pi,size=(samples,N_axions))
        else:
            range_initial = (2*np.pi)/np.amin(np.abs(Q)[np.nonzero(Q)])
            theta_0 = np.random.uniform(0,range_initial,size=(samples,N_axions))
        
        theta_min = np.zeros_like(theta_0)
        grad_in = np.zeros_like(theta_0)
        Grad_min = np.zeros_like(theta_0)
    
        for s in range(samples):
            grad_in[s,:] = Nabla(theta_0[s,:],Q,L_log,L_sign,phases) #test with positive Lambdas
            theta_min[s,:] = optimize.root(Nabla, x0 = theta_0[s,:], args=(Q,L_log,L_sign,phases), jac=Hess, method='hybr').x #lm #hybr
            Grad_min[s,:] = Nabla(theta_min[s,:] % (2*np.pi) ,Q,L_log,L_sign,phases)
    
        #### WE DO SOME FILTERING ON THE ARRAYS ####
        theta_mask_0 = np.where(np.abs(theta_min) > 1e-05, theta_min, 0)
        mask_small_grad = np.all(np.abs(Grad_min) < 1e-5,axis=1) #1e-5
        theta_small_grad_nc = theta_mask_0[mask_small_grad] #Table of minima
        Grad_min_small_nc = Grad_min[mask_small_grad] #Table Gradient at minima
        n_extrema = np.shape(theta_small_grad_nc)[0]
    
        min_list = []
        for m in range(n_extrema):
            if np.amin(np.linalg.eigh(Hessian_true(theta_small_grad_nc[m,:],Q,L_log,L_sign,phases))[0]) > 0:
                min_list = np.append(min_list,m)
    
        ###### POSITION OF MINIMA (PROBABLY REPEATED) FOUND IN REDUECED SYSTEM ######
        min_list = np.array(min_list)
        theta_min_nc = theta_small_grad_nc[min_list.astype(int),:]
        number_min = theta_min_nc.shape
    
    
        # FROM OUR LIST OF MINIMA WE TAKE OUT THETA_TILDE_VECTORS WHICH ARE ALL ZEROS
        pos_non_zero = np.where(np.asarray(np.count_nonzero(theta_min_nc,axis=1, keepdims=True)) >= 1)[0] #very important
        theta_non_zero = theta_min_nc[pos_non_zero]
        Norms = [np.linalg.norm(theta_non_zero[y,:], np.inf) for y in range(0,np.shape(theta_non_zero)[0])] #Norm_inf
        order_Norm = np.array(Norms).argsort()
        Norm_Thetas = np.sort(Norms)
        Thetas_total = np.around(theta_non_zero[order_Norm,:],6)
    
        ###### 	ALGORITHM TO GET UNIQUE MINIMA ######
        if Z == 1: #ALL alpha-vectors have integer entries so we mod by 2pi
            Thetas_total_mod = Thetas_total % (2*np.pi) #position of minima between 0 and 2pi this helps to distinguish them!
            theta_mask = np.where(np.abs(Thetas_total_mod) > 1e-04, Thetas_total_mod, 0)
            theta_mask_2pi = np.where(np.abs(np.abs(theta_mask)-2*np.pi) > 1e-04, theta_mask, 0)
            theta_mask_pi = np.where(np.abs(np.abs(theta_mask_2pi) - np.pi) > 1e-04, theta_mask_2pi, np.pi)
    
            ###### WE FIND THE UNIQUE VECTORS IN THE LIST ######
            Theta_unique_mod = np.unique(theta_mask_pi,axis=0)
            Theta_round = np.where(Theta_unique_mod==np.pi , Theta_unique_mod, np.around(Theta_unique_mod, decimals=4)) #decimals=6
            Theta_unique_final = np.unique(Theta_round, axis=0)
    
            #TOTAL MIN FOR INTEGER ALPHA-VECTORS
            N_unique = Theta_unique_final.shape[0]
            Total_min_theory = np.abs(det_QTilde * N_unique)
        else:	#at least one entry has a rational number
            c_vector = np.matmul(inv_symmetries,Thetas_total.T / (2*np.pi)).T
            c_vector_mod = c_vector.copy()
            c_vector_mod = c_vector_mod % 1
            
            ###### WE FIND THE UNIQUE VECTORS IN THE LIST ######
            c_vector_mask = np.where(np.abs(c_vector_mod) > 1e-04, c_vector_mod, 0)
            c_vector_mask_1 = np.where(np.abs(np.abs(c_vector_mask)- 1) > 1e-04, c_vector_mask, 0)
    
            ###### WE FIND THE UNIQUE VECTORS IN THE LIST ######
            c_vector_unique_mod = np.unique(c_vector_mask_1,axis=0)
            c_vector_round = np.around(c_vector_unique_mod, decimals=4)
            c_vector_unique_final = np.unique(c_vector_round, axis=0)
    
            #WE CONSRTUCT THE VECTOR OF THETA-MIN
            symmetries = np.where(np.abs(np.linalg.inv(inv_symmetries)) > 1e-04, np.linalg.inv(inv_symmetries), 0)
            Theta_unique_final = np.matmul(symmetries,c_vector_unique_final.T * (2*np.pi)).T
    
            #TOTAL MIN FOR RATIONAL ALPHA-VECTORS
            N_unique = Theta_unique_final.shape[0]
            Total_min_theory = np.abs((det_QTilde/det_Sym) * N_unique)
    
        return Total_min_theory, Theta_unique_final, det_QTilde
        """
end

function one_dim_axion_solver(T::Solver1D)
    search_domain = T.search_domain
    Q = T.Q
    Llog = T.Llog
    Lsign = T.Lsign 
    det_QTilde = T.det_QTilde
    phases = T.phases
    Z = T.Z
    inv_symmetries = T.inv_symmetries
    det_Sym = T.det_Sym
    Total_min_theory, Theta_unique_final, det_QTilde = py"one_dim_axion_solver($search_domain, $Q, $Llog, $Lsign, $det_QTilde, $phases, $Z, $inv_symmetries, $det_Sym)"
    Min_JLM_1D(Total_min_theory, Theta_unique_final, 1, Int(det_QTilde))
end

function one_dim_axion_solver(search_domain, Q, Llog, Lsign, det_QTilde, phases, Z, inv_symmetries, det_Sym)
    Total_min_theory, Theta_unique_final = py"one_dim_axion_solver($search_domain, $Q, $Llog, $Lsign, $det_QTilde, $phases, $Z, $inv_symmetries, $det_Sym)"
    Min_JLM_1D(Total_min_theory, Theta_unique_final, 1, Int(det_QTilde))
end

function multi_axion_solver(T::SolverND)
    samples = T.samples
    Q = T.Q
    Llog = T.Llog
    Lsign = T.Lsign 
    det_QTilde = T.det_QTilde
    phases = T.phases
    Z = T.Z
    inv_symmetries = T.inv_symmetries
    det_Sym = T.det_Sym
    Total_min_theory, Theta_unique_final = py"multi_axion_solver($samples, $Q, $Llog, $Lsign, $det_QTilde, $phases, $Z, $inv_symmetries, $det_Sym)"
    Min_JLM_ND(Total_min_theory, Theta_unique_final, size(Q, 1) - size(Q, 2), Int(det_QTilde))
end

function multi_axion_solver(samples, Q, Llog, Lsign, det_QTilde, phases, Z, inv_symmetries, det_Sym)
    Total_min_theory, Theta_unique_final = py"multi_axion_solver($samples, $Q, $Llog, $Lsign, $det_QTilde, $phases, $Z, $inv_symmetries, $det_Sym)"
    Min_JLM_ND(Total_min_theory, Theta_unique_final, size(Q, 1) - size(Q, 2), Int(det_QTilde))
end

end