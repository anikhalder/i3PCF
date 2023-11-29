import numpy as np
import halo_utils
from scipy.integrate import quad
import constants

# matter projection kernels

def q_m(z, n_m_z_func, H_z_func):
    '''
    x : chi (comoving distance)
    q_m(x) = n_m(z) dz/dx
    => q_m(x(z)) = n_m(z) H(z)
    '''
    return n_m_z_func(z)*H_z_func(z)

# convergence projection kernels

def q_k_zs_fixed(z, zs, chi_z_func, H_0, Omega0_m):
    '''
    x : chi (comoving distance) ; x' : chi' (comoving distance to source redshift) ; denote z' as zs
    q_k(x) = 3/2 H0^2 Omega0_m x/a(x) (x'-x)/x'
    => q_k(x(z)) = 3/2 H0^2 Omega0_m (1+z) x(z) (x(z')-x(z))/x(z')
    '''
    if (z > zs):
        return 0

    chi_z = chi_z_func(z)
    chi_zs = chi_z_func(zs)

    return 3./2. * H_0 * H_0 * Omega0_m * (1.+z) * chi_z * (chi_zs - chi_z) / chi_zs

def W_k_zs_distribution_integrand(zs, z, chi_z_func, n_s_z_func, delta_photoz=0.0):
    '''
    dz'/H(z') g(x(z')) (x(z')-x(z))/x(z')
    = dz'/H(z') n_source(z') H(z') (x(z')-x(z))/x(z') , where g(x') = n_source(z') H(z')
    = dz' n_source(z') (x(z')-x(z))/x(z')
    Here, denote z' as zs -- is the variable of the intergation
    '''

    if (z > zs):
        return 0

    chi_z = chi_z_func(z)
    chi_zs = chi_z_func(zs)

    return n_s_z_func(zs + delta_photoz) * (chi_zs - chi_z) / chi_zs

def q_k_zs_distribution(z, zs_max, chi_z_func, H_0, Omega0_m, n_s_z_func):
    '''
    equations (10.42) and (10.41) combined of https://edoc.ub.uni-muenchen.de/23401/1/Friedrich_Oliver.pdf
    or equations (6.21) and (6.19) combined of https://arxiv.org/pdf/astro-ph/9912508.pdf

    x : chi (comoving distance) ; x' : chi' (comoving distance to source redshift) ; denote z' as zs
    q_k(x) = 3/2 H0^2 Omega0_m x/a(x) W_k(x) , where W_k(x) = \int_x^x_max dx'g(x') (x'-x)/x'
    => q_k(x) = 3/2 H0^2 Omega0_m x/a(x) \int_x^x_max dx'g(x') (x'-x)/x'
    => q_k(x(z)) = 3/2 H0^2 Omega0_m (1+z) x(z) \int_z^z_max dz'/H(z') g(x(z')) (x(z')-x(z))/x(z')
    => q_k(x(z)) = 3/2 H0^2 Omega0_m (1+z) x(z) W_k_zs_distribution(x(z))
    '''

    chi_z = chi_z_func(z)
    #W_k_zs_distribution_z = quad(W_k_zs_distribution_integrand, z, zs_max, args=(z, chi_z_func, n_s_z_func))[0]

    #zs_array = np.linspace(z, zs_max, 100)
    zs_array = np.arange(z, zs_max, 0.01)
    chi_zs_array = chi_z_func(zs_array)
    W_k_zs_distribution_z_array = n_s_z_func(zs_array) * (chi_zs_array - chi_z) / chi_zs_array
    W_k_zs_distribution_z = np.trapz(W_k_zs_distribution_z_array, x=zs_array)
   
    return 3./2. * H_0 * H_0 * Omega0_m * (1.+z) * chi_z * W_k_zs_distribution_z

def q_k_zs_distribution_systematics(z, zs_max, chi_z_func, H_0, Omega0_m, n_s_z_func, H_z_func, D_plus_z_func, A_IA_0_NLA, alpha_IA_0_NLA, delta_photoz, m):
    '''
    equations (10.42) and (10.41) combined of https://edoc.ub.uni-muenchen.de/23401/1/Friedrich_Oliver.pdf
    or equations (6.21) and (6.19) combined of https://arxiv.org/pdf/astro-ph/9912508.pdf

    x : chi (comoving distance) ; x' : chi' (comoving distance to source redshift) ; denote z' as zs
    q_k(x) = 3/2 H0^2 Omega0_m x/a(x) W_k(x) , where W_k(x) = \int_x^x_max dx'g(x') (x'-x)/x'
    => q_k(x) = 3/2 H0^2 Omega0_m x/a(x) \int_x^x_max dx'g(x') (x'-x)/x'
    => q_k(x(z)) = 3/2 H0^2 Omega0_m (1+z) x(z) \int_z^z_max dz'/H(z') g(x(z')) (x(z')-x(z))/x(z')
    => q_k(x(z)) = 3/2 H0^2 Omega0_m (1+z) x(z) W_k_zs_distribution(x(z))
    '''

    chi_z = chi_z_func(z)
    #W_k_zs_distribution_z = quad(W_k_zs_distribution_integrand, z, zs_max, args=(z, chi_z_func, n_s_z_func, delta_photoz))[0]

    #zs_array = np.linspace(z, zs_max, 100)
    zs_array = np.arange(z, zs_max, 0.01)
    chi_zs_array = chi_z_func(zs_array)
    W_k_zs_distribution_z_array = n_s_z_func(zs_array + delta_photoz) * (chi_zs_array - chi_z) / chi_zs_array
    W_k_zs_distribution_z = np.trapz(W_k_zs_distribution_z_array, x=zs_array)
   
    q_k_z = 3./2. * H_0 * H_0 * Omega0_m * (1.+z) * chi_z * W_k_zs_distribution_z

    if (A_IA_0_NLA == 0.0):
        return (1+m)*q_k_z
    else:
        f_IA_NLA_z = - A_IA_0_NLA * 0.0134 * Omega0_m * ((1.+z)/1.62)**alpha_IA_0_NLA * D_plus_z_func(0)/D_plus_z_func(z)
        return (1+m)*(q_k_z + f_IA_NLA_z * H_z_func(z) * n_s_z_func(z + delta_photoz))

# Halo projection kernels

def N_h_z_integrand(z, z_min, z_max, M_min, M_max, rho0_m, sigma_R_z_func, sigma_prime_R_z_func, chi_z_func, H_z_func):

    return 1/H_z_func(z)*q_h_bias_unnormalised(z, z_min, z_max, M_min, M_max, 'n_h', rho0_m, sigma_R_z_func, sigma_prime_R_z_func, chi_z_func)

def N_h(z_min, z_max, M_min, M_max, rho0_m, sigma_R_z_func, sigma_prime_R_z_func, chi_z_func, H_z_func):
    '''
    Total number of halos inside a given mass bin [M_min, M_max] over the whole comoving volume between [z_min, z_max]
    M_min, M_max must be in [M_sun] units
    rho0_m must be in units of [M_sun/Mpc^3]
    '''

    return quad(N_h_z_integrand, z_min, z_max, args=(z_min, z_max, M_min, M_max, rho0_m, sigma_R_z_func, sigma_prime_R_z_func, chi_z_func, H_z_func))[0]

def q_h_bias_unnormalised(z, z_min, z_max, M_min, M_max, bias_type, rho0_m, sigma_R_z_func, sigma_prime_R_z_func, chi_z_func):
    '''
    eqns (11) and (15) of https://iopscience.iop.org/article/10.3847/1538-4357/aa943d/pdf
    M, M_min, M_max must be in [M_sun] units
    rho0_m must be in units of [M_sun/Mpc^3]
    '''

    if (z < z_min or z > z_max):
        return 0.

    chi_z = chi_z_func(z)

    if (bias_type == '1_over_n_h'):
        return constants._f_sky_*4.*np.pi*chi_z*chi_z # this is an unnormalised quantity --> divide by the total number of halos in bin, N_h, to normalise
    else:
        return constants._f_sky_*4.*np.pi*chi_z*chi_z*halo_utils.bias_h_z_unnormalised(z, M_min, M_max, bias_type, rho0_m, sigma_R_z_func, sigma_prime_R_z_func) # this is an unnormalised quantity --> divide by the total number of halos in bin, N_h, to normalise