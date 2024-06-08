from scipy.optimize import fsolve

## function to compute individual masses of massive neutrinos

def get_masses(sum_masses, hierarchy, delta_m_squared_atm=2.5e-3, delta_m_squared_sol=7.37e-5):
    """ Return individual masses of neutrinos

    Args:
        sum_masses ( float ): sum of neutrino masses
        hierarchy ( string ): Neutrino hierarchy
        delta_m_squared_atm ( float ): squared difference of neutrino masses from atmospheric oscillation experiments
        delta_m_squared_sol ( float ): squared difference of neutrino masses from solar oscillation experiments
    Returns:
        m1, m2, m3: Individual neutrino masses
    """

    if (sum_masses == 0.0):
        # massless neutrino case
        m1 = 0.0
        m2 = 0.0
        m3 = 0.0
        return m1, m2, m3
    
    if (hierarchy == 'DEGENERATE'):
        # degenerate neutrino case
        m1 = sum_masses / 3
        m2 = sum_masses / 3
        m3 = sum_masses / 3
        return m1, m2, m3
    
    elif (hierarchy == 'NORMAL'):
        # delta_m_squared_atm = m_3^2 - (m_1^2 + m_2^2)/2
        # delta_m_squared_sol = m_2^2 - m_1^2
        m1_func = lambda m1, M_tot, d_m_sq_atm, d_m_sq_sol: M_tot**2. + 0.5*d_m_sq_sol - d_m_sq_atm + m1**2. - 2.*M_tot*m1 - 2.*M_tot*(d_m_sq_sol+m1**2.)**0.5 + 2.*m1*(d_m_sq_sol+m1**2.)**0.5
        m1, opt_output, success, output_message = fsolve(m1_func, sum_masses/3.,(sum_masses,delta_m_squared_atm,delta_m_squared_sol), full_output=True)
        m1 = m1[0]
        m2 = (delta_m_squared_sol + m1**2.)**0.5
        m3 = (delta_m_squared_atm + 0.5*(m2**2. + m1**2.))**0.5
        return m1, m2, m3
    
    elif (hierarchy == 'INVERTED'):
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
    
    else:
        raise ValueError('Unknown hierarchy %s'%hierarchy)