import numpy as np
from scipy.special import j1

######################################################################################

## 2D window functions

def spherical_cap_sqdeg_2_radius(patch_area_sq_degrees):
    return np.arccos(1-patch_area_sq_degrees*np.pi/(2.0*180*180))

def spherical_cap_radius_2_sqradians(theta_T):
    # patch is basically a spherical cap centered around the North Pole on a sphere of unit radius
    r = 1.0
    return 2.*np.pi*r*r*(1.-np.cos(theta_T))

def W2D_TH_RS(length_arr, theta_T):
    # normalised 2D tophat (real space)
    val = np.zeros(length_arr.size)
    val[length_arr <= theta_T] = 1.0/(np.pi*theta_T*theta_T)
    return val

def W2D_TH_RS_unnormalised(length_arr, theta_T):
    # normalised 2D tophat (real space)
    val = np.zeros(length_arr.size)
    val[length_arr <= theta_T] = 1.0
    return val

def W2D_TH_FS(l_arr, theta_T):
    # normalised 2D tophat (Fourier space)
    val = np.ones(l_arr.size)
    mask = (l_arr != 0.0)
    x = l_arr*theta_T
    val[mask] = j1(x[mask])/(x[mask])*2 #*np.pi*theta_T*theta_T # unnormalised
    return val

def W2D_U_RS(length_arr, theta_T):
    # see eqn (10) of https://www.aanda.org/articles/aa/pdf/2005/40/aa3531-05.pdf
    x = length_arr*length_arr
    y = 1.0/(2.0*theta_T*theta_T)
    return y/np.pi * (1-x*y)*np.exp(-x*y)

def W2D_U_FS(l_arr, theta_ap):
    # see the line below eqn (11) of https://www.aanda.org/articles/aa/pdf/2005/40/aa3531-05.pdf
    x = l_arr*theta_ap
    return 0.5*x*x * np.exp(-0.5*x*x)

######################################################################################

## 3D window functions

def W3D_TH_FS(k_arr, R):
    # normalised 3D tophat (Fourier space)
    # see eqn (A.114) of Dodelson & Schmidt (2020)
    # k_arr is an array of k [1/Mpc] and R [Mpc] is the size of tophat --> both are in comoving coordinates
    x = k_arr*R
    return -3./(x*x*x)*(-x*np.cos(x) + np.sin(x))

def W3D_prime_TH_FS(k_arr, R):
    # derivative of normalised 3D tophat wrt R (Fourier space)
    # d W(kR) / d R = (d W(kR) / d kR)*(d kR / d R) = k * (d W(kR) / d kR)
    # k_arr is an array of k [1/Mpc] and R [Mpc] is the size of tophat --> both are in comoving coordinates
    x = k_arr*R
    return k_arr*(-9./(x*x*x*x)*(-x*np.cos(x) + np.sin(x)) + 3.*np.sin(x)/(x*x))