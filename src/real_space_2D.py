import numpy as np
from scipy import interpolate
from scipy.special import lpn
from scipy.special import lpmn
import constants
from cosmology_utils import T17_resolution_correction

def C_ell_spherical_sky(l_array, P_l_array):
    '''
    In this function, we use the following notation:
    l_array and P_l_array array are computed for 2D flat sky at 2D Fourier l values until some maximum l_max (which can be a floating number)
    ell and C_ell are computed for 2D spherical sky at every integer multipole ells = 0,1,...,l_max-1
    '''

    # compute interpolated C_ells array for ells = 0,1,...,l_max-1

    P_l_interpolated = interpolate.interp1d(l_array, P_l_array, fill_value=(0,0), bounds_error=False)

    ell = np.arange(0, int(l_array[-1])-1)
    C_ell = np.zeros(ell.size)

    mask = (ell != 0) & (ell != 1)
    l = ell[mask]

    # Kitching++ correction
    C_ell[mask] = (l+2.)*(l+1.)*l*(l-1.) / ((l+0.5)**4) * P_l_interpolated(l+0.5)

    if (constants._apply_T17_corrections_ == True):
        C_ell *= T17_resolution_correction(ell)

    return ell, C_ell

def xi_theta(theta, ell, C_ell):
    '''
    theta is in radians
    assume ell starts from 0 and the values of C_ell at the first two multipoles (i.e. ell=0,1) are zero
    eqn (9) of Friedrich++ 2020 DES covariance modelling --- https://arxiv.org/pdf/2012.08568.pdf
    '''

    l = ell[2:] # take the values from ell=2,...,ell_max for the summation below
    C_l = C_ell[2:]

    x = np.cos(theta)

    P_ell = lpn(ell[-1],x)[0] # all the (first kind) Legendre polynomials of degree ell, P_ell values from ell=0,1,2,...ell_max
    P_l = P_ell[2:] # take the values from ell=2,...,ell_max for the summation below

    return np.sum( (2.*l+1) / (4*np.pi) * P_l * C_l )

def xit_theta(theta, ell, C_ell):
    '''
    theta is in radians
    assume ell starts from 0 and the values of C_ell at the first two multipoles (i.e. ell=0,1) are zero
    eqn (9) of Friedrich++ 2020 DES covariance modelling --- https://arxiv.org/pdf/2012.08568.pdf
    '''

    l = ell[2:] # take the values from ell=2,...,ell_max for the summation below
    C_l = C_ell[2:]

    x = np.cos(theta)

    P_ell_2 = lpmn(2,ell[-1],x)[0][-1] # all the associated Legendre polynomials of degree ell and order m=2, P_ell^2 values from ell=0,1,2,...,ell_max
    P_l_2 = P_ell_2[2:] # take the values from ell=2,...,ell_max for the summation below

    return np.sum( (2.*l+1) / (4*np.pi) * P_l_2 / (l*(l+1)) * C_l )

def xip_theta(theta, ell, C_ell):
    '''
    theta is in radians    
    assume ell starts from 0 and the values of C_ell at the first two multipoles (i.e. ell=0,1) are zero
    eqn (9) of Friedrich++ 2020 DES covariance modelling --- https://arxiv.org/pdf/2012.08568.pdf
    eqn (A5) of Friedrich++ 2020 DES covariance modelling --- https://arxiv.org/pdf/2012.08568.pdf
    here, G_l_2_x_p = G_l_2_+(x) + G_l_2_-(x) i.e. LHS of Friedrich++ for the 'p'lus middle sign
    '''

    l = ell[2:] # take the values from ell=2,...,ell_max for the summation below
    C_l = C_ell[2:]

    x = np.cos(theta)

    P_ell_2 = lpmn(2,ell[-1],x)[0][-1] # all the associated Legendre polynomials of degree ell and order m=2, P_ell^2 values from ell=0,1,2,...,ell_max
    P_l_2 = P_ell_2[2:] # take the values from ell=2,...,ell_max for the summation below
    P_lm1_2 = P_ell_2[1:-1] # take the values from ell=1,...,ell_max-1 for the summation below

    G_l_2_x_p = P_l_2*( (4.-l+2.*x*(l-1.))/(1.-x*x) - l*(l-1.)/2. ) + P_lm1_2*( (l+2.)*(x-2.)/(1.-x*x))

    return np.sum( (2.*l+1) / (4*np.pi) * 2. * G_l_2_x_p / (l*l*(l+1.)*(l+1.)) * C_l )

def xim_theta(theta, ell, C_ell):
    '''
    theta is in radians 
    assume ell starts from 0 and the values of C_ell at the first two multipoles (i.e. ell=0,1) are zero
    eqn (9) of Friedrich++ 2020 DES covariance modelling --- https://arxiv.org/pdf/2012.08568.pdf
    eqn (A5) of Friedrich++ 2020 DES covariance modelling --- https://arxiv.org/pdf/2012.08568.pdf
    here, G_l_2_x_m = G_l_2_+(x) - G_l_2_-(x) i.e. LHS of Friedrich++ for the 'm'inus middle sign
    '''

    x = np.cos(theta)

    l = ell[2:] # take the values from ell=2,...,ell_max for the summation below
    C_l = C_ell[2:]

    P_ell_2 = lpmn(2,ell[-1],x)[0][-1] # all the P_l^2 values from ell=0,1,2,...ell_max

    P_l_2 = P_ell_2[2:] # take the values from ell=2,...,ell_max for the summation below
    P_lm1_2 = P_ell_2[1:-1] # take the values from ell=1,...,ell_max-1 for the summation below

    G_l_2_x_m = P_l_2*( (4.-l-2.*x*(l-1.))/(1.-x*x) - l*(l-1.)/2. ) + P_lm1_2*( (l+2.)*(x+2.)/(1.-x*x))

    return np.sum( (2.*l+1) / (4*np.pi) * 2. * G_l_2_x_m / (l*l*(l+1.)*(l+1.)) * C_l )

def xi_theta_bin_averaged_Legendre(theta_min, theta_max, ell, C_ell):
    '''
    theta_min and theta_max are in radians
    assume ell starts from 0 and the values of C_ell at the first two multipoles (i.e. ell=0,1) are zero
    eqn (9) of Friedrich++ 2020 DES covariance modelling --- https://arxiv.org/pdf/2012.08568.pdf
    eqn (65) of Friedrich++ 2020 DES covariance modelling --- https://arxiv.org/pdf/2012.08568.pdf
    '''

    l = ell[2:] # take the values from ell=2,...,ell_max for the summation below
    C_l = C_ell[2:]

    x_min = np.cos(theta_min)
    x_max = np.cos(theta_max)

    P_ell_x_min_array = lpn(ell[-1]+1,x_min)[0] # all the (first kind) Legendre polynomials of degree ell, P_ell values from ell=0,1,2,...ell_max
    P_lp1_x_min_array = P_ell_x_min_array[3:] # take the values from ell=3,...,ell_max+1 for the summation below
    P_lm1_x_min_array = P_ell_x_min_array[1:-2] # take the values from ell=1,...ell_max-1 for the summation below

    P_ell_x_max_array = lpn(ell[-1]+1,x_max)[0]
    P_lp1_x_max_array = P_ell_x_max_array[3:]
    P_lm1_x_max_array = P_ell_x_max_array[1:-2]

    P_l_bin_averaged = 1./(2.*l+1.)/(x_min - x_max)*(P_lp1_x_min_array-P_lm1_x_min_array-(P_lp1_x_max_array-P_lm1_x_max_array))

    return P_l_bin_averaged
    #return np.sum( (2.*l+1) / (4*np.pi) * P_l_bin_averaged * C_l )

def xit_theta_bin_averaged_Legendre(theta_min, theta_max, ell, C_ell):
    '''
    theta_min and theta_max is in radians
    assume ell starts from 0 and the values of C_ell at the first two multipoles (i.e. ell=0,1) are zero
    eqn (9) of Friedrich++ 2020 DES covariance modelling --- https://arxiv.org/pdf/2012.08568.pdf
    eqn (B2) of Friedrich++ 2020 DES covariance modelling --- https://arxiv.org/pdf/2012.08568.pdf
    '''

    l = ell[2:] # take the values from ell=2,...,ell_max for the summation below
    C_l = C_ell[2:]

    x_min = np.cos(theta_min)
    x_max = np.cos(theta_max)

    P_ell_x_min_array = lpn(ell[-1]+1,x_min)[0] # all the (first kind) Legendre polynomials of degree ell, P_ell values from ell=0,1,2,...ell_max
    P_lp1_x_min_array = P_ell_x_min_array[3:] # take the values from ell=3,...,ell_max+1 for the summation below
    P_l_x_min_array = P_ell_x_min_array[2:-1] # take the values from ell=2,...,ell_max for the summation below
    P_lm1_x_min_array = P_ell_x_min_array[1:-2] # take the values from ell=1,...ell_max-1 for the summation below

    P_ell_x_max_array = lpn(ell[-1]+1,x_max)[0] # all the (first kind) Legendre polynomials of degree ell, P_ell values from ell=0,1,2,...ell_max
    P_lp1_x_max_array = P_ell_x_max_array[3:] 
    P_l_x_max_array = P_ell_x_max_array[2:-1] 
    P_lm1_x_max_array = P_ell_x_max_array[1:-2] 

    P_l_2_bin_averaged = 1./(x_min - x_max)*(
                        (l+2./(2*l+1))*(P_lm1_x_min_array-P_lm1_x_max_array)
                        +(2.-l)*(x_min*P_l_x_min_array-x_max*P_l_x_max_array)
                        -(2./(2*l+1))*(P_lp1_x_min_array-P_lp1_x_max_array)
                        )

    return P_l_2_bin_averaged
    #return np.sum( (2.*l+1) / (4*np.pi) * P_l_2_bin_averaged / (l*(l+1)) * C_l )

def xip_theta_bin_averaged_Legendre(theta_min, theta_max, ell, C_ell):
    '''
    theta_min and theta_max is in radians    
    assume ell starts from 0 and the values of C_ell at the first two multipoles (i.e. ell=0,1) are zero
    eqn (9) of Friedrich++ 2020 DES covariance modelling --- https://arxiv.org/pdf/2012.08568.pdf
    eqn (B5) of Friedrich++ 2020 DES covariance modelling --- https://arxiv.org/pdf/2012.08568.pdf
    here, G_l_2_x_p = G_l_2_+(x) + G_l_2_-(x) i.e. LHS of Friedrich++ for the 'p'lus middle sign
    '''

    l = ell[2:] # take the values from ell=2,...,ell_max for the summation below
    C_l = C_ell[2:]

    x_min = np.cos(theta_min)
    x_max = np.cos(theta_max)

    P_ell_x_min_array, ddx_P_ell_x_min_array = lpn(ell[-1]+1,x_min) # all the (first kind) Legendre polynomials and derivatives of degree ell, P_ell values from ell=0,1,2,...ell_max
    P_lp1_x_min_array = P_ell_x_min_array[3:] # take the values from ell=3,...,ell_max+1 for the summation below
    P_l_x_min_array = P_ell_x_min_array[2:-1] # take the values from ell=2,...,ell_max for the summation below
    P_lm1_x_min_array = P_ell_x_min_array[1:-2] # take the values from ell=1,...ell_max-1 for the summation below
    ddx_P_l_x_min_array = ddx_P_ell_x_min_array[2:-1] 
    ddx_P_lm1_x_min_array = ddx_P_ell_x_min_array[1:-2] 

    P_ell_x_max_array, ddx_P_ell_x_max_array = lpn(ell[-1]+1,x_max) # all the (first kind) Legendre polynomials and derivatives of degree ell, P_ell values from ell=0,1,2,...ell_max
    P_lp1_x_max_array = P_ell_x_max_array[3:] # take the values from ell=3,...,ell_max+1 for the summation below
    P_l_x_max_array = P_ell_x_max_array[2:-1] # take the values from ell=2,...,ell_max for the summation below
    P_lm1_x_max_array = P_ell_x_max_array[1:-2] # take the values from ell=1,...ell_max-1 for the summation below
    ddx_P_l_x_max_array = ddx_P_ell_x_max_array[2:-1] 
    ddx_P_lm1_x_max_array = ddx_P_ell_x_max_array[1:-2] 

    G_l_2_x_p_bin_averaged = 1./(x_min - x_max)*(
                            -l*(l-1.)/2.*(l+2./(2.*l+1.))*(P_lm1_x_min_array-P_lm1_x_max_array)
                            -l*(l-1.)*(2.-l)/2.*(x_min*P_l_x_min_array-x_max*P_l_x_max_array)
                            +l*(l-1.)/(2.*l+1.)*(P_lp1_x_min_array-P_lp1_x_max_array)
                            +(4.-l)*(ddx_P_l_x_min_array-ddx_P_l_x_max_array)
                            +(l+2.)*(x_min*ddx_P_lm1_x_min_array-x_max*ddx_P_lm1_x_max_array-(P_lm1_x_min_array-P_lm1_x_max_array))
                            +2.*(l-1.)*(x_min*ddx_P_l_x_min_array-x_max*ddx_P_l_x_max_array-(P_l_x_min_array-P_l_x_max_array))
                            -2.*(l+2.)*(ddx_P_lm1_x_min_array-ddx_P_lm1_x_max_array)
                             )

    return G_l_2_x_p_bin_averaged
    #return np.sum( (2.*l+1) / (4*np.pi) * 2. * G_l_2_x_p_bin_averaged / (l*l*(l+1.)*(l+1.)) * C_l )

def xim_theta_bin_averaged_Legendre(theta_min, theta_max, ell, C_ell):
    '''
    theta_min and theta_max is in radians    
    assume ell starts from 0 and the values of C_ell at the first two multipoles (i.e. ell=0,1) are zero
    eqn (9) of Friedrich++ 2020 DES covariance modelling --- https://arxiv.org/pdf/2012.08568.pdf
    eqn (B5) of Friedrich++ 2020 DES covariance modelling --- https://arxiv.org/pdf/2012.08568.pdf
    here, G_l_2_x_m = G_l_2_+(x) - G_l_2_-(x) i.e. LHS of Friedrich++ for the 'm'inus middle sign
    '''

    l = ell[2:] # take the values from ell=2,...,ell_max for the summation below
    C_l = C_ell[2:]

    x_min = np.cos(theta_min)
    x_max = np.cos(theta_max)

    P_ell_x_min_array, ddx_P_ell_x_min_array = lpn(ell[-1]+1,x_min) # all the (first kind) Legendre polynomials and derivatives of degree ell, P_ell values from ell=0,1,2,...ell_max
    P_lp1_x_min_array = P_ell_x_min_array[3:] # take the values from ell=3,...,ell_max+1 for the summation below
    P_l_x_min_array = P_ell_x_min_array[2:-1] # take the values from ell=2,...,ell_max for the summation below
    P_lm1_x_min_array = P_ell_x_min_array[1:-2] # take the values from ell=1,...ell_max-1 for the summation below
    ddx_P_l_x_min_array = ddx_P_ell_x_min_array[2:-1] 
    ddx_P_lm1_x_min_array = ddx_P_ell_x_min_array[1:-2] 

    P_ell_x_max_array, ddx_P_ell_x_max_array = lpn(ell[-1]+1,x_max) # all the (first kind) Legendre polynomials and derivatives of degree ell, P_ell values from ell=0,1,2,...ell_max
    P_lp1_x_max_array = P_ell_x_max_array[3:] # take the values from ell=3,...,ell_max+1 for the summation below
    P_l_x_max_array = P_ell_x_max_array[2:-1] # take the values from ell=2,...,ell_max for the summation below
    P_lm1_x_max_array = P_ell_x_max_array[1:-2] # take the values from ell=1,...ell_max-1 for the summation below
    ddx_P_l_x_max_array = ddx_P_ell_x_max_array[2:-1] 
    ddx_P_lm1_x_max_array = ddx_P_ell_x_max_array[1:-2] 

    G_l_2_x_m_bin_averaged = 1./(x_min - x_max)*(
                            -l*(l-1.)/2.*(l+2./(2.*l+1.))*(P_lm1_x_min_array-P_lm1_x_max_array)
                            -l*(l-1.)*(2.-l)/2.*(x_min*P_l_x_min_array-x_max*P_l_x_max_array)
                            +l*(l-1.)/(2.*l+1.)*(P_lp1_x_min_array-P_lp1_x_max_array)
                            +(4.-l)*(ddx_P_l_x_min_array-ddx_P_l_x_max_array)
                            +(l+2.)*(x_min*ddx_P_lm1_x_min_array-x_max*ddx_P_lm1_x_max_array-(P_lm1_x_min_array-P_lm1_x_max_array))
                            -2.*(l-1.)*(x_min*ddx_P_l_x_min_array-x_max*ddx_P_l_x_max_array-(P_l_x_min_array-P_l_x_max_array))
                            +2.*(l+2.)*(ddx_P_lm1_x_min_array-ddx_P_lm1_x_max_array)
                             )

    return G_l_2_x_m_bin_averaged
    #return np.sum( (2.*l+1) / (4*np.pi) * 2. * G_l_2_x_m_bin_averaged / (l*l*(l+1.)*(l+1.)) * C_l )