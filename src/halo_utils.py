import numpy as np
import constants
from scipy.integrate import quad

def R_L(M, rho0_m):
    '''
    comoving Lagrangian radius [Mpc] of a halo of mass M [M_sun]
    Note: as this is a comoving quantity it doesn't matter at which redshift z the halo is identified
    physical radius of halo at z : R_halo(z) = (3M/(4 pi rho(z)))^(1/3)
                                             = (3M a(z)^3/(4 pi rho_0))^(1/3)
                                             = a(z) (3M/(4 pi rho_0))^(1/3)
                                            := a(z) R_L
    ==> R_L = (3M / (4 pi rho_0) )^(1/3) in comoving [Mpc] e.g. see eqn (12.63) of Dodelson & Schmidt (2020)

    Note: rho0_m must be in units of [M_sun/Mpc^3]
    '''

    return (3.0*M/(4.0*np.pi*rho0_m))**(1./3.) # in comoving [Mpc]

def halo_b1_PBS(M, z, rho0_m, sigma_R_z_func):
    '''
    Eulerian b1 bias in the Peak Background Split model using the Press Schechter halo mass function
    '''
    # see eqn (2) of Jing++ (1998) https://iopscience.iop.org/article/10.1086/311530/pdf
    R = R_L(M, rho0_m)
    nu = constants._delta_c_ / sigma_R_z_func(R, z)
    b1_L = (nu*nu-1.)/constants._delta_c_ # Lagrangian b1 bias

    return 1. + b1_L; # Eulerian b1 bias

def halo_b2_PBS(M, z, rho0_m, sigma_R_z_func):
    '''
    Eulerian b2 bias in the Peak Background Split model using the Press Schechter halo mass function
    '''
    R = R_L(M, rho0_m)
    nu = constants._delta_c_ / sigma_R_z_func(R, z)
    b1_L = (nu*nu-1.)/constants._delta_c_ # Lagrangian b1 bias

    nu2 = nu*nu
    b2_L = nu2*(nu2-3.)/(constants._delta_c_*constants._delta_c_); # Lagrangian b2 bias

    return 8./21.*b1_L + b2_L; # Eulerian b2 bias (the 8/21 factor appears from spherical symmetric collapse)

def halo_b2_Lazeyras(M, z, rho0_m, sigma_R_z_func):
    '''
    Eulerian b2 bias with the fitting function of Lazeyras++ 
    '''
    #b1 = halo_b1_PBS(M, z, rho0_m, sigma_R_z_func) # b1 PBS bias
    b1 = halo_b1_Tinker2010(M, z, rho0_m, sigma_R_z_func) # b1 Tinker bias

    return 0.412 - 2.143*b1 + 0.929*b1*b1 + 0.008*b1*b1*b1

def halo_bs2_coevolution(M, z, rho0_m, sigma_R_z_func):
    '''
    Eulerian tidal bs2 bias from coevolution relation
    e.g. see Table 1 Leicht++ 2020 and also Pandey++ 2020
    '''
    #b1 = halo_b1_PBS(M, z, rho0_m, sigma_R_z_func) # b1 PBS bias
    b1 = halo_b1_Tinker2010(M, z, rho0_m, sigma_R_z_func) # b1 Tinker bias

    return -4./7.*(b1-1); # Note that in Desjacques++ and many other literature one works with bK2 = 1/2*bs2 = -2/7(b1-1)

def halo_b1_nu_Tinker2010(nu):
    '''
    Eulerian b1 bias (Tinker fitting function) of a halo of peak height nu
    eqn (6) and Table 2 of Tinker++ (2010) https://iopscience.iop.org/article/10.1088/0004-637X/724/2/878/pdf
    '''

    y = np.log10(constants._Delta_)
    xp = np.exp(-1.0*(4./y)**4)

    A = 1. + 0.24*y*xp
    a = 0.44*y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107*y + 0.19*xp
    c = 2.4

    nu_a = nu**a
    nu_b = nu**b
    nu_c = nu**c

    return 1. - A*nu_a/(nu_a+constants._delta_c_**a) + B*nu_b + C*nu_c
    
def halo_b1_Tinker2010(M, z, rho0_m, sigma_R_z_func):
    '''
    Eulerian b1 bias (Tinker fitting function) of a halo of mass M [M_sun]
    eqn (6) and Table 2 of Tinker++ (2010) https://iopscience.iop.org/article/10.1088/0004-637X/724/2/878/pdf
    '''
    R = R_L(M, rho0_m)
    nu = constants._delta_c_ / sigma_R_z_func(R, z)

    return halo_b1_nu_Tinker2010(nu)

def f_nu_PS(nu):
    '''
    Press Schechter HMF expressed with peak height nu
    eqn (12.73) of Dodelson & Schmidt (2020)
    '''
    return np.sqrt(2./np.pi)*nu*np.exp(-nu*nu*0.5)

def dn_dM_PS(M, z, rho0_m, sigma_R_z_func, sigma_prime_R_z_func):
    '''
    Press Schechter HMF in units of peak height nu
    halo mass M is in units [M_sun]
    eqn (12.73) of Dodelson & Schmidt (2020)

    dn/dln M = (rho_0/M)*f_PS(nu)* |dln sigma(M,z) / dln M|    where nu = delta_c / sigma(M,z)
    dn/dM = (rho_0/M)*f_PS(nu)* |dln sigma(M,z) / d M|
          = (rho_0/M)*f_PS(nu)* (dln sigma^-1 / d M)
          = (rho_0/M)*f_PS(nu)*(1/(3M))* (dln sigma^-1 / dln R)    where dM = 3M dln R
          = f_PS(nu)*(-1/3)*(rho_0/M^2)* (dln sigma / dln R)
          = f_PS(nu)*(-1/3)*(rho_0/M^2)* R/sigma (d sigma / d R)

    dn/dln M = M dn/dM = f_PS(nu)*(-1/3)*(rho_0/M)* R/sigma (d sigma / d R)
    '''

    R = R_L(M, rho0_m)
    sigma_R_z = sigma_R_z_func(R, z)
    sigma_prime_R_z = sigma_prime_R_z_func(R, z) # in [1/Mpc]

    nu = constants._delta_c_ / sigma_R_z

    f_PS = f_nu_PS(nu)

    return f_PS*(-1./3.)*(rho0_m/M**2)*R/sigma_R_z*sigma_prime_R_z # in [1/Mpc^3/M_sun]

def f_nu_Tinker2010(nu, z):
    '''
    Tinker HMF expressed with peak height nu
    eqn (8) and Table 4 of Tinker++ (2010) https://iopscience.iop.org/article/10.1088/0004-637X/724/2/878/pdf

    Note: 'f_nu' of Tinker is defined differently from 'f_nu' of PS. There is a factor of nu which differs one from the other besides the functional form itself.
    '''

    assert(constants._Delta_ == 200)

    # these parameters were calibrated at z=0
    alpha_200 = 0.368
    beta_200 = 0.589
    gamma_200 = 0.864
    phi_200 = -0.729
    eta_200 = -0.243

    alpha = alpha_200
    beta = beta_200*(1.+z)**0.20
    phi = phi_200*(1.+z)**-0.08
    eta = eta_200*(1.+z)**0.27
    gamma = gamma_200*(1.+z)**-0.01

    f_nu_Tinker = alpha*(1. + (beta*nu)**(-2*phi))*nu**(2*eta)*np.exp(-gamma*nu*nu*0.5)

    return f_nu_Tinker

def g_sigma_Tinker2010(sigma_R_z, z):
    '''
    eqn (8) and Table 4 of Tinker++ (2010) https://iopscience.iop.org/article/10.1088/0004-637X/724/2/878/pdf
    '''

    nu = constants._delta_c_ / sigma_R_z

    return nu*f_nu_Tinker2010(nu, z)

def f_sigma_Tinker2008(sigma_R_z, z):
    '''
    eqn (3) and Table 2 of Tinker++ (2008) https://iopscience.iop.org/article/10.1086/591439/pdf
    '''

    assert(constants.__Delta__ == 200)

    A0_200 = 0.186
    a0_200 = 1.47
    b0_200 = 2.57
    c0_200 = 1.19

    A = A0_200*(1.+z)**-0.14
    a = a0_200*(1.+z)**-0.06
    alpha = 10**(-(0.75/(np.log10(constants._Delta_/75.0))**1.2))
    b = b0_200*(1.+z)**-alpha
    c = c0_200

    f_sigma = A*((sigma_R_z/b)**-a + 1.)*np.exp(-c/sigma_R_z**2)

    return f_sigma

def dn_dM_Tinker2008(M, z, rho0_m, sigma_R_z_func, sigma_prime_R_z_func):
    '''
    halo mass M is in units [M_sun]
    eqn (2) of Tinker++ (2008) https://iopscience.iop.org/article/10.1086/591439/pdf

    dn/dM = f(sigma)*(rho_0/M)* (dln sigma^-1 / d M)
          = f(sigma)*(rho_0/M)*(1/(3M))* (dln sigma^-1 / dln R)    where dM = 3M dln R
          = f(sigma)*(-1/3)*(rho_0/M^2)* (dln sigma / dln R)
          = f(sigma)*(-1/3)*(rho_0/M^2)* R/sigma (d sigma / d R)

    dn/dln M = M dn/dM = f(sigma)*(-1/3)*(rho_0/M)* R/sigma (d sigma / d R)
    '''
    
    R = R_L(M, rho0_m)
    sigma_R_z = sigma_R_z_func(R, z)
    sigma_prime_R_z = sigma_prime_R_z_func(R, z) # in [1/Mpc]

    #f_sigma = f_sigma_Tinker2008(sigma_R_z, z)
    f_sigma = g_sigma_Tinker2010(sigma_R_z, z)

    return f_sigma*(-1./3.)*(rho0_m/M**2)*R/sigma_R_z*sigma_prime_R_z # in [1/Mpc^3/M_sun]

def HMF_M_integrand(M, z, weight_type, rho0_m, sigma_R_z_func, sigma_prime_R_z_func):

    if (weight_type == 'n_h'):
        W_M_z = 1.
    elif (weight_type == 'b1'):
        W_M_z = halo_b1_Tinker2010(M, z, rho0_m, sigma_R_z_func)
    elif (weight_type == 'b2'):
        W_M_z = halo_b2_Lazeyras(M, z, rho0_m, sigma_R_z_func)
    elif (weight_type == 'bs2'):
        W_M_z = halo_bs2_coevolution(M, z, rho0_m, sigma_R_z_func)

    return dn_dM_Tinker2008(M, z, rho0_m, sigma_R_z_func, sigma_prime_R_z_func)*W_M_z

def n_h_z(z, M_min, M_max, rho0_m, sigma_R_z_func, sigma_prime_R_z_func):
    '''
    3D number density of halos [1/Mpc^3] at a given redshift z inside a given mass bin [M_min, M_max]
    M_min, M_max must be in [M_sun] units
    rho0_m must be in units of [M_sun/Mpc^3]
    '''

    return quad(HMF_M_integrand, M_min, M_max, args=(z, 'n_h', rho0_m, sigma_R_z_func, sigma_prime_R_z_func))[0]

def bias_h_z_unnormalised(z, M_min, M_max, bias_type, rho0_m, sigma_R_z_func, sigma_prime_R_z_func):
    '''
    effective UNNORMALISED bias of halos at a given redshift z inside a given mass bin [M_min, M_max]
    M_min, M_max must be in [M_sun] units
    rho0_m must be in units of [M_sun/Mpc^3]
    '''

    # this is an unnormalised quantity --> divide by the 3D number density of halos in mass bin, n_h_z, to normalise
    return quad(HMF_M_integrand, M_min, M_max, args=(z, bias_type, rho0_m, sigma_R_z_func, sigma_prime_R_z_func))[0] 

def bias_h_z(z, M_min, M_max, bias_type, rho0_m, sigma_R_z_func, sigma_prime_R_z_func):
    '''
    effective bias of halos at a given redshift z inside a given mass bin [M_min, M_max]
    M_min, M_max must be in [M_sun] units
    rho0_m must be in units of [M_sun/Mpc^3]
    '''

    return bias_h_z_unnormalised(z, M_min, M_max, bias_type, rho0_m, sigma_R_z_func, sigma_prime_R_z_func) / n_h_z(z, M_min, M_max, rho0_m, sigma_R_z_func, sigma_prime_R_z_func)

# HOD functions

def N_galaxies_expected_HOD(M):
    # halo mass M is in units [M_sun]

    #'''
    # this is using the Lens1-Source4 RedMagic best fit parameters from Table D1 of Zacharegkas 2020 (DES collaboration)
    log_M_min = 12.25 #12.3
    M_1 = pow(10,13.5)
    sigma_logM = 0.50
    alpha = 2.06
    f_cen = 0.2
    ###log_M_cut = 12.0413926852
    #'''

    '''
    # this is using the Lens3-Source4 MagLim best fit parameters from Table D2 of Zacharegkas 2020 (DES collaboration)
    sigma_log_M_min = 0.09
    sigma_M_1 = 0.31
    sigma_sigma_logM = 0.14
    sigma_alpha = 0.24

    log_M_min = 11.88 + 1.7*sigma_log_M_min
    M_1 = pow(10,(12.84 + 1.7*sigma_M_1))
    sigma_logM = 0.21 - 1.4*sigma_sigma_logM
    alpha = 1.24 - 3.0*sigma_alpha
    f_cen = 1.0

    # the above is the same as
    #log_M_min = 12.033
    #M_1 = pow(10,13.37)
    #sigma_logM = 0.014
    #alpha = 0.52
    #f_cen = 1.0
    ###log_M_cut = 12.7075701761
    '''

    # eqn (1) and (2) of Zacharegkas 2020

    N_cen_expected = f_cen/2.0*(1.+np.erf((np.log10(M) - log_M_min)/sigma_logM))
    N_sat_expected = N_cen_expected*(M/M_1)**alpha

    return N_cen_expected + N_sat_expected;
    #return N_cen_expected;
    #return N_sat_expected;