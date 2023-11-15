import numpy as np 
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import fsolve

######################################################################################

## function to compute individual masses of massive neutrinos

def get_masses(delta_m_squared_atm, delta_m_squared_sol, sum_masses, hierarchy):
    """ Return individual masses of neutrinos

    Args:
        delta_m_squared_atm ( float ): squared difference of neutrino masses from atmospheric oscillation experiments
        delta_m_squared_sol ( float ): squared difference of neutrino masses from solar oscillation experiments
        sum_masses ( float ): sum of neutrino masses
        hierarchy ( string ): Neutrino hierarchy

    Returns:
        m1, m2, m3: Individual neutrino masses
    """

    if (sum_masses == 0.0):
        # massless neutrino case
        m1 = 0.0
        m2 = 0.0
        m3 = 0.0
        return m1, m2, m3
    
    # any string containing letter 'n' will be considered as refering to normal hierarchy
    if ('n' in hierarchy.lower()):
        # delta_m_squared_atm = m_3^2 - (m_1^2 + m_2^2)/2
        # delta_m_squared_sol = m_2^2 - m_1^2
        m1_func = lambda m1, M_tot, d_m_sq_atm, d_m_sq_sol: M_tot**2. + 0.5*d_m_sq_sol - d_m_sq_atm + m1**2. - 2.*M_tot*m1 - 2.*M_tot*(d_m_sq_sol+m1**2.)**0.5 + 2.*m1*(d_m_sq_sol+m1**2.)**0.5
        m1, opt_output, success, output_message = fsolve(m1_func, sum_masses/3.,(sum_masses,delta_m_squared_atm,delta_m_squared_sol), full_output=True)
        m1 = m1[0]
        m2 = (delta_m_squared_sol + m1**2.)**0.5
        m3 = (delta_m_squared_atm + 0.5*(m2**2. + m1**2.))**0.5
        return m1, m2, m3
    else:
        # Inverted hierarchy massive neutrinos. Calculates the individual
        # neutrino masses from M_tot_IH and deletes M_tot_IH
        #delta_m_squared_atm = -m_3^2 + (m_1^2 + m_2^2)/2
        #delta_m_squared_sol = m_2^2 - m_1^2
        delta_m_squared_atm = -delta_m_squared_atm
        m1_func = lambda m1, M_tot, d_m_sq_atm, d_m_sq_sol: M_tot**2. + 0.5*d_m_sq_sol - d_m_sq_atm + m1**2. - 2.*M_tot*m1 - 2.*M_tot*(d_m_sq_sol+m1**2.)**0.5 + 2.*m1*(d_m_sq_sol+m1**2.)**0.5
        m1, opt_output, success, output_message = fsolve(m1_func, sum_masses/3.,(sum_masses,delta_m_squared_atm,delta_m_squared_sol),full_output=True)
        m1 = m1[0]
        m2 = (delta_m_squared_sol + m1**2.)**0.5
        m3 = (delta_m_squared_atm + 0.5*(m2**2. + m1**2.))**0.5
        return m1, m2, m3
    
######################################################################################

## function to compute density parameter of massive neutrinos

def Mnu2omeganu(sum_masses, Omega_m_0, Omega_b_0, h):
    # the atmospheric and solar neutrino mass constraints are from the MassiveNuS paper by Liu et al.
    m1, m2, m3 = get_masses(2.5e-3, 7.37e-5, sum_masses, 'NH') 
    mnu_arr = np.array([m1, m2, m3])* u.eV
    cosmo = FlatLambdaCDM(H0=h*100, Om0=Omega_m_0, Neff=3.046, m_nu=mnu_arr, Ob0=Omega_b_0, Tcmb0=2.7255)
    return cosmo.Onu0*(h**2)