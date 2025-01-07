#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from scipy.special import comb as comb
import matplotlib.pyplot as plt

from typing import List, Tuple, Union
import warnings

import fs_rdp_bounds as fsrdp


def K_Wang(alpha,sigma):
    K_terms=[1.]
    alpha_prod=alpha*(alpha-1)
    
    K_terms.append(2*alpha_prod*q**2*(np.exp(4/sigma**2)-1))
    
    for j in range(3,alpha+1):
        alpha_prod=alpha_prod*(alpha-j+1)
        K_terms.append(2*q**j*alpha_prod/math.factorial(j)*np.exp((j-1)*2*j/sigma**2))

    K=0
    for j in range(len(K_terms)):
        K=K+K_terms[len(K_terms)-1-j] 
    K=np.log(K)
    return K

def Wang_et_al_upper_bound(alpha,sigma):
    if alpha>=2:
        if int(alpha)==alpha:
            return 1./(alpha-1.)*K_Wang(alpha,sigma)
        else:
            return (1.-(alpha-math.floor(alpha)))/(alpha-1)*K_Wang(math.floor(alpha),sigma)+(alpha-math.floor(alpha))/(alpha-1)*K_Wang(math.floor(alpha)+1,sigma)
    else:
        return Wang_et_al_upper_bound(2,sigma)

def Wang_et_al_lower_bound(alpha,sigma):
    if int(alpha)==alpha:
        L_terms=[1.]
        L_terms.append(alpha*q/(1-q))
        alpha_prod=alpha
        for j in range(2,alpha+1):
            alpha_prod=alpha_prod*(alpha-j+1)
            L_terms.append(alpha_prod/np.math.factorial(j)*(q/(1-q))**j*np.exp((j-1)*2*j/sigma**2))
        
        
        L=0
        for j in range(len(L_terms)):
            L=L+L_terms[len(L_terms)-1-j]         
        return alpha/(alpha-1)*np.log(1-q)+1/(alpha-1)*np.log(L)
    else:
        print("Error, alpha must be an integer.")


def get_eps(*, orders: Union[List[float], float], rdp
            : Union[List[float], float], delta: float) -> Tuple[float, float]:
    r"""Computes epsilon given a list of Renyi Differential Privacy (RDP) values at
    multiple RDP orders and target ``delta``.
    The computation of epslion, i.e. conversion from RDP to (eps, delta)-DP,
    is based on the theorem presented in the following work:
    Borja Balle et al. "Hypothesis testing interpretations and Renyi differential privacy."
    International Conference on Artificial Intelligence and Statistics. PMLR, 2020.
    Particullary, Theorem 21 in the arXiv version https://arxiv.org/abs/1905.09982.
    Args:
        orders: An array (or a scalar) of orders (alphas).
        rdp: A list (or a scalar) of RDP guarantees.
        delta: The target delta.
    Returns:
        Pair of epsilon and optimal order alpha.
    Raises:
        ValueError
            If the lengths of ``orders`` and ``rdp`` are not equal.
    """
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)

    if len(orders_vec) != len(rdp_vec):
        raise ValueError(
            f"Input lists must have the same length.\n"
            f"\torders_vec = {orders_vec}\n"
            f"\trdp_vec = {rdp_vec}\n"
        )

    eps = (
        rdp_vec
        - (np.log(delta) + np.log(orders_vec)) / (orders_vec - 1)
        + np.log((orders_vec - 1) / orders_vec)
    )

    # special case when there is no privacy
    if np.isnan(eps).all():
        return np.inf, np.nan

    idx_opt = np.nanargmin(eps)  # Ignore NaNs
    if idx_opt == 0 or idx_opt == len(eps) - 1:
        extreme = "smallest" if idx_opt == 0 else "largest"
        warnings.warn(
            f"Optimal order is the {extreme} alpha. Please consider expanding the range of alphas to get a tighter privacy bound. delta is: {delta}"
        )
    return eps[idx_opt], orders_vec[idx_opt]

def return_epsilon(sigma,mode,dataset_size):
    n_train = dataset_size
    batch_size = 550
    q=batch_size/n_train
    r_over_sigma_tilde=2./sigma
    N_alpha=500
    alpha_array=1+10**np.linspace(-1,1.5,N_alpha)
    m_array=[4]
    N_m=len(m_array)
    delta = 1/dataset_size
    n_epochs = 20

    accu_factor = n_train / batch_size * n_epochs

        
    ### plot guarantees
    if(mode == "our"):
        eps_array=np.zeros((N_m,N_alpha))
        for j1 in range(N_m):
            for j2 in range(N_alpha):
                eps_array[j1,j2]=fsrdp.FSwoR_RDP_ro(alpha_array[j2], sigma, m_array[j1], q)
        rdp = eps_array[0, :] * accu_factor
        ep, alpha = get_eps(orders = alpha_array, rdp = rdp, delta = delta)
    elif(mode == "wang"):
        Wang_eps_array=np.zeros(N_alpha)
        for j1 in range(N_alpha):
            Wang_eps_array[j1]=Wang_et_al_upper_bound(alpha_array[j1],sigma)
        rdp = Wang_eps_array * accu_factor
        ep, alpha = get_eps(orders = alpha_array, rdp = rdp, delta = delta)
    elif(mode == "rdp"):
        eps_array_poisson = np.zeros((N_m,N_alpha))
        for j1 in range(N_m):
            for j2 in range(N_alpha):
                eps_array_poisson[j1,j2]=fsrdp.Poisson_RDP_ro(alpha_array[j2], sigma, m_array[j1], q)
        rdp = eps_array_poisson[0, :] * accu_factor
        ep, alpha = get_eps(orders = alpha_array, rdp = rdp, delta = delta)

    # print("Poisson Subsampled RDP {}".format(ep))
    return ep

def find_proper_noise(target_epsilon,mode,max_noise,steps,dataset_size):
    epsilon_map = {}
    diffs = []
    for i in range(1,max_noise*steps):
        noise = i * (1/steps)
        diff = abs(target_epsilon-return_epsilon(noise,mode,dataset_size))
        diffs.append(diff)
        epsilon_map[diff] = noise
    min_value = min(diffs)
    with open("kasra_experment.txt", "a") as f:
        f.write(f"noise should be {epsilon_map[min_value]}, for epsilon {target_epsilon}, mode is {mode}, diff is {min_value}, data_set size is: {dataset_size}\n")
    print("noise should be {}, for epsilon {}, mode is {}, diff is {}".format(epsilon_map[min_value],target_epsilon,mode,min_value))

modes = ["wang","our","rdp"]
target_epsilons = [10]
dataset_size_list = [3491,3357,2244,2159,5869]
for mode in modes:
    for epsilon in target_epsilons:
        for dataset in dataset_size_list:
            find_proper_noise(target_epsilon=epsilon,mode=mode,max_noise=5,steps=100,dataset_size=dataset)



        
    
    
