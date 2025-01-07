#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:23:18 2024

"""

##Implementation of fixed-size subsampling (with and without replacement) Renyi Differential Privacy (FS-RDP) method from the paper
##Differentially Private Stochastic Gradient Descent with Fixed-Size Minibatches: Tighter RDP Guarantees with or without Replacement

#abbreviations:
#wR: subsampling with replacement
#woR: subsampling without replacement

#ar: bounds under add/remove adjacency definition
#ro: bounds under replace-one adjacency definition
import math
import numpy as np
from scipy.special import comb as comb




def log_comb(k,n):
    val=0
    for j in range(n):
        val=val+np.log(k-j)
    for j in range(1,n+1):
        val=val-np.log(j)
    return val

def log_factorial(k):
    val=0
    for j in range(1,k):
        val=val+np.log(j+1)
    return val



#Definition of auxiliary functions needed for all of the bounds 
def log_M_exact(sigma,k):
    if k==0:
        return 0
    elif k>=2:
        exponents=[]
        
        exponents.append(np.log(k-1))
        for ell in range(2,k+1):
            exponents.append(log_comb(k,ell)+2.*ell*(ell-1)/sigma**2)
            
        exp_max=np.max(exponents)
        
        
        M=(-1)**(k-1)*np.exp(exponents[0]-exp_max)
        j=1
        for ell in range(2,k+1):
            M=M+(-1)**(k-ell)*np.exp(exponents[j]-exp_max)
            j=j+1
        return np.log(M)+exp_max
    
def M_exact(sigma,k):
    return np.exp(log_M_exact(sigma,k))
    

def log_B_bound(sigma,m):
    if m%2==0:
        return log_M_exact(sigma,m)
    else:
        return 0.5*(log_M_exact(sigma,m-1)+log_M_exact(sigma,m+1))

def B_bound(sigma,m):
    if m%2==0:
        return M_exact(sigma,m)
    else:
        return M_exact(sigma,m-1)**(1/2)*M_exact(sigma,m+1)**(1/2)
    

    
def log_B_bound_precomputed(m,log_M_exact_2j_array):
    if m%2==0:
        return log_M_exact_2j_array[int(m/2)]
    else:
        return 0.5*(log_M_exact_2j_array[int((m-1)/2)]+log_M_exact_2j_array[int((m+1)/2)])

def B_bound_precomputed(m,log_M_exact_2j_array):
    return np.exp(log_B_bound_precomputed(m,log_M_exact_2j_array))


##Functions for computing FS-woR  bounds
#assumes that alpha\neq j for j={1,...,m-1}: this condition should be checked whenever this function is used
def log_R_bound(alpha,sigma,m,q):
    log_abs_alpha_prod=np.log(alpha)
    for j in range(1,m):
        log_abs_alpha_prod=log_abs_alpha_prod+np.log(np.abs(alpha-j))
        
    if alpha-m<=0:
        return np.log(q)*m-log_factorial(m)+np.log(1-q)*(alpha-m)+log_abs_alpha_prod+log_B_bound(sigma,m)
    
    
    else:
        alpha_ceil=np.ceil(alpha)
        
        exponents=[]
        
        exponents.append(log_B_bound(sigma,m)-log_factorial(m))
        for ell in range(int(alpha_ceil)-m+1):
            
            log_fact_ratio=0
            for j in range(ell):
                log_fact_ratio=log_fact_ratio+np.log(alpha_ceil-m-j)
                
            
            exponents.append(np.log(q)*ell+log_fact_ratio-log_factorial(m+ell)+log_B_bound(sigma,ell+m))
        exp_max=np.max(exponents)
        return np.log(q)*m+log_abs_alpha_prod+np.log(np.sum(np.exp(exponents-exp_max)))+exp_max
    
def R_bound(alpha,sigma,m,q):
    if alpha==int(alpha) and alpha<m:
        return 0
    
    else:
        return np.exp(log_R_bound(alpha,sigma,m,q))

    

def H_bound(alpha,sigma,m,q):
    H_terms=[1.]
    log_alpha_prod=np.log(alpha)
    s=1
    for k in range(2,m):
        log_alpha_prod=log_alpha_prod+np.log(np.abs(alpha-k+1))
        s=s*np.sign(alpha-k+1)
        H_terms.append(s*np.exp(np.log(q)*k-log_factorial(k)+log_alpha_prod+log_M_exact(sigma,k)))
    H_terms.append(R_bound(alpha,sigma,m,q))
    

    return np.sum(H_terms)


def log_H_bound_minus_1(alpha,sigma,m,q):
    exponents=[]
    signs=[]
    s=1
    
    log_alpha_prod=np.log(alpha)
    for k in range(2,m):
        if alpha-k+1==0: #no further terms contribute
            break  
        else:
            log_alpha_prod=log_alpha_prod+np.log(np.abs(alpha-k+1))
            s=s*np.sign(alpha-k+1)
            signs.append(s)
            exponents.append(np.log(q)*k-log_factorial(k)+log_alpha_prod+log_M_exact(sigma,k))
        
    if not(alpha==int(alpha) and alpha<m): #remainder is zero if alpha is an integer in {1,...,m-1}
        signs.append(1)
        exponents.append(log_R_bound(alpha,sigma,m,q))
    
    exp_max=np.max(exponents)

    return np.log(np.sum(signs*np.exp(exponents-exp_max)))+exp_max




#One step RDP bound for fixed-size subsampling without replacement under add/remove adjacency (Thm 3.3 with T=1)
def FSwoR_RDP_ar(alpha,sigma,m,q):
    assert m==int(m) and m>=3, "m must be an integer greater than or equal to 3"
    assert q>0 and q<1, "q must be between in (0,1)"
    assert alpha>1, "alpha must be greater than 1"
    assert sigma>0, "sigma must be greater than zero"

    val=log_H_bound_minus_1(alpha,sigma,m,q)
    if val<=0:
        return 1/(alpha-1.)*np.log(1+np.exp(val)) 
    else:
        return 1/(alpha-1.)*(np.log(np.exp(-val)+1)+val)
        
    
#Thm 3.1 bounds for integer alpha (Taylor remainder is zero)
def K_integer_alpha_ar(alpha,sigma,q):
    K_terms=[]
    for k in range(alpha+1):
        K_term=comb(alpha, k, exact=True)*(1-q)**(alpha-k)*q**k*np.exp(k*(k-1)*(2./sigma)**2/2)
        K_terms.append(K_term)
    
    
    
    #Sum terms in reverse order for numerical stability
    K=0
    for j in range(len(K_terms)):
        K=K+K_terms[len(K_terms)-1-j] 
    K=np.log(K)
    return K


## FS_woR RDP bounds under replace-one adjacency (Thm 3.4 bounds)

#upper bound on log of kth-derivative for k>=3
def log_F_deriv_bound(alpha,sigma,k,log_M_exact_2j_array):
    exponents=[]
    if k%2==0:
        exponents.append(np.log(4)+log_B_bound_precomputed(k,log_M_exact_2j_array))
    else:
        exponents.append(np.log(3)+log_B_bound_precomputed(k,log_M_exact_2j_array))
        
    
    
    for j in range(k+1):
        prod1=1.
        for ell in range(1,j):
            prod1=prod1*(1-ell/alpha)
        prod2=1.
        for ell in range(k-j):
            prod2=prod2*(1+(ell-1)/alpha)
        
        C=np.abs((alpha/(alpha-1))*prod1*prod2-1.)
        
        if C>0:  
            exponents.append(log_comb(k,j)+np.log(C)+log_B_bound_precomputed(k,log_M_exact_2j_array))
        
        exp_max=np.max(exponents)
            
        val=np.log(np.sum(np.exp(exponents-exp_max)))+exp_max
    return np.log(alpha-1.)+np.log(alpha)*(k-1)+val


    
def E_bound(alpha,sigma,m,q,log_M_exact_2j_array):
    val=0
    
    for j in range(m+1):
        if not(alpha==int(alpha) and alpha<j): #otherwise this j doesn't contribure
            sum1=0.
            for ell in range(j):                
                sum1=sum1+np.log(np.abs(alpha-ell))
            sum2=0.
            for ell in range(m-j):
                sum2=sum2+np.log(alpha+ell-1)
                
            
            if alpha-j<=0:                                   
                I=np.log(1-q)*(alpha-j)+log_B_bound_precomputed(m,log_M_exact_2j_array)
            
            else:
                val2=B_bound_precomputed(m,log_M_exact_2j_array)
                alpha_ceil=np.ceil(alpha)
                for ell in range(int(alpha_ceil)-j+1):
                    D=0
                    for ell2 in range(ell):
                        D=D+np.log((alpha_ceil-j-ell2)/(m+ell-ell2))
                    
                    val2=val2+np.exp(ell*np.log(q)+D+log_B_bound_precomputed(ell+m,log_M_exact_2j_array))
                
                I=np.log(val2)
            
            val=val+np.exp(np.log(q)*m-log_factorial(m)-(alpha+m-j-1)*np.log(1-q)+log_comb(m,j)+sum1+sum2+I)
        
        
    return val


def FSwoR_RDP_ro(alpha,sigma,m,q):
    assert m==int(m) and m>=3, "m must be an integer greater than or equal to 3"
    assert q>0 and q<1, "q must be between in (0,1)"
    assert alpha>1, "alpha must be greater than 1"
    assert sigma>0, "sigma must be greater than zero"

    #precompute required M_exact values for speed
    if (int(np.ceil(alpha))+m)%2==0:
        n_precompute=int((m+np.ceil(alpha))/2)+1
    else:
        n_precompute=int((m+np.ceil(alpha)+1)/2)+1

    n_precompute=int(n_precompute)    

    log_M_exact_2j_array=np.zeros(n_precompute)
    for j in range(n_precompute):
        log_M_exact_2j_array[j]=log_M_exact(sigma,2*j)
   
    
    expansion_terms=1.+q**2*alpha*(alpha-1)*(np.exp(4./sigma**2)-np.exp(2./sigma**2))
    for k in range(3,m):
        expansion_terms=expansion_terms+np.exp(np.log(q)*k-log_factorial(k)+log_F_deriv_bound(alpha,sigma,k,log_M_exact_2j_array))


    
    return 1./(alpha-1)*np.log(expansion_terms+E_bound(alpha,sigma,m,q,log_M_exact_2j_array))

#One step RDP bound for Poisson subsampling under replace-one adjacency (Theorem C.9)
def Poisson_RDP_ro(alpha,sigma,m,q):
    assert m==int(m) and m>=3, "m must be an integer greater than or equal to 3"
    assert q>0 and q<1, "q must be in (0,1)"
    assert alpha>1, "alpha must be greater than 1"
    assert sigma>0, "sigma must be greater than zero"

    #precompute required M_exact values for speed
    if (int(np.ceil(alpha))+m)%2==0:
        n_precompute=int((m+np.ceil(alpha))/2)+1
    else:
        n_precompute=int((m+np.ceil(alpha)+1)/2)+1

    n_precompute=int(n_precompute)    

    log_M_exact_2j_array=np.zeros(n_precompute)
    for j in range(n_precompute):
        log_M_exact_2j_array[j]=log_M_exact(2*sigma,2*j)
   
    
    expansion_terms=1.+q**2*alpha*(alpha-1)*(np.exp(1./sigma**2)-np.exp(-1./sigma**2))
    for k in range(3,m):
        expansion_terms=expansion_terms+np.exp(np.log(q)*k-log_factorial(k)+log_F_deriv_bound(alpha,2*sigma,k,log_M_exact_2j_array))


    
    return 1./(alpha-1)*np.log(expansion_terms+E_bound(alpha,2*sigma,m,q,log_M_exact_2j_array))


##One step RDP bound for fixed-size subsampling with replacement under add/remove adjacency (Theorem 3.6)
#K is the number of terms that are not bounded using the worst-case result
def FSwR_RDP_ar(alpha,sigma,m,B,D,K=None):
    assert m==int(m) and m>=3, "m must be an integer greater than or equal to 3"
    assert B/D>0 and B/D<1, "B/D must be in (0,1)"
    assert alpha>1, "alpha must be greater than 1"
    assert sigma>0, "sigma must be greater than zero"

    
    if K is None:
        K=np.minimum(m-1,B)
        
    #compute q_tilde
    q_tilde=0.
    for n in range(1,K+1):
        q_tilde=q_tilde+np.exp(log_comb(B,n)+np.log(D)*(-n)+np.log(1.-1./D)*(B-n))
    q_tilde=1./(1.+(1.-1./D)**B/q_tilde)

    exponents=[0.]
    for n in range(1,K+1):
        log_a_n= log_comb(B,n)+np.log(D)*(-n)+np.log(1.-1./D)*(B-n)
        log_a_n_tilde=log_a_n-np.log(q_tilde)
        tmp=log_H_bound_minus_1(alpha,sigma/n,m,q_tilde)
        #take minimum of Taylor expansion bound and worst-case bound of each term
        tmp=np.minimum(tmp,np.log(1.-np.exp(-2*alpha*(alpha-1)*n**2/sigma**2))+2*alpha*(alpha-1)*n**2/sigma**2)
        exponents.append(log_a_n_tilde+tmp)
    for n in range(K+1,B+1):
        log_a_n= log_comb(B,n)+np.log(D)*(-n)+np.log(1.-1./D)*(B-n)
        tmp=np.log(1.-np.exp(-2*alpha*(alpha-1)*n**2/sigma**2))+2*alpha*(alpha-1)*n**2/sigma**2
        exponents.append(log_a_n+tmp)

    exp_max=np.max(exponents)
    

    
    return 1./(alpha-1.)*(np.log(np.sum(np.exp(exponents-exp_max)))+exp_max)




#Lower bound on FS-RDP with replacement from Theorem 3.7 and Appendix D.2.1.
    
#lb computed using the relaxation from Appendix D.2.1: indices_function(k) should return indices of terms used in stage k of the recursion
def log_FSwR_ar_LB_inductive_approx(k,c,d,B,D, indices_function):
    if k==2:
        log_F_terms=[]
        indices_k=indices_function(k)
        for n in indices_k:
            log_a_n=log_comb(B,n)-n*np.log(D)+(B-n)*np.log(1-1/D)
            
            log_F_terms.append(log_a_n+d*n+B*np.log((1-1/D)*np.exp(-(c*n+d))+D**(-1))+B*(c*n+d))
       
        max_log_F=np.max(log_F_terms)    
        F_terms=np.exp(log_F_terms-max_log_F)

        return np.log(np.sum(F_terms))+max_log_F
    else:
        log_F_terms=[]
        indices_k=indices_function(k)
        for n in indices_k:
            log_F_terms.append(log_comb(B,n)+np.log(D)*(-n)+np.log(1-1/D)*(B-n)+d*n+log_FSwR_ar_LB_inductive_approx(k-1,c,d+c*n,B,D,indices_function))
        max_log_F=np.max(log_F_terms)    
        
        F_terms=np.exp(log_F_terms-max_log_F)

        return np.log(np.sum(F_terms))+max_log_F
    
#FSR lower bound computed using selected indices
def FSwR_RDP_ar_LB_approximate(alpha,sigma,B,D,indices_function):
    assert B/D>=0 and B/D<=1, "B/D must be between 0 and 1"
    assert alpha>1, "alpha must be greater than 1"
    assert sigma>0, "sigma must be greater than zero"
    
    return 1/(alpha-1)*log_FSwR_ar_LB_inductive_approx(alpha,4/sigma**2,0,B,D,indices_function)


#exact formula for the Theorem 3.7 lower bound on FS-RDP with replacement  (don't use if unless |B| and alpha are small!)
#exact lb for alpha=2
def FSwR_ar_LB_2_exact(c,d,B,D):
    log_F_terms=[]
   
    for n in range(B+1):
        log_a_n=log_comb(B,n)-n*np.log(D)+(B-n)*np.log(1-1/D)
        
        log_F_terms.append(log_a_n+d*n+B*np.log((1-1/D)*np.exp(-(c*n+d))+D**(-1))+B*(c*n+d))
   
    max_log_F=np.max(log_F_terms)    
    F_terms=np.exp(log_F_terms-max_log_F)

    return np.log(np.sum(F_terms))+max_log_F

def FSwR_ar_LB_inductive_exact(k,c,d,B,D):
    if k==2:
        return np.exp(FSwR_ar_LB_2_exact(c,d,B,D))
    else:
        F=0
        for n in range(B+1):
            F=F+comb(B, n, exact=True)*D**(-n)*(1-1/D)**(B-n)*np.exp(d*n)*FSwR_ar_LB_inductive_exact(k-1,c,d+c*n,B,D)
        return F

def FSwR_RDP_ar_LB_exact(alpha,sigma,B,D):
    assert B/D>=0 and B/D<=1, "B/D must be between 0 and 1"
    assert alpha>1, "alpha must be greater than 1"
    assert sigma>0, "sigma must be greater than zero"
    
    return 1/(alpha-1)*np.log(FSwR_ar_LB_inductive_exact(alpha,4/sigma**2,0,B,D))




