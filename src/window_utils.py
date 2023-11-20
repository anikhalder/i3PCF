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


######################################################################################

'''

## area pre-factors

def iZ_A2pt(params):
    # Function to compute the iB(l,z) at a given (l,z) grid point with vegas integration
    # Wse polar coordinates here where p[2] and p[3] are polar angles (radian) of 2D wavevctors l1 and l2.
    # Without loss of generality can always assume that vector l coincides with the positive x-axis i.e. phi_l = 0.
    alpha_min = params[0]
    alpha_max = params[1]
    theta_T = params[2]

    integ = vegas.Integrator([[alpha_min, alpha_max], [0, theta_T], [0, 2*np.pi]]) # l1, l2, phi_1, phi_2

    if (B3D_type == 'lin'):
        diB_app_l_z = diB_l_z_batch(l, z, "iB_UWWp", theta_U, theta_T, CosmoClassObject, "B_tree")
    elif (B3D_type == 'nl'):
        if (input.iB_app_type == 'GM'):
            diB_app_l_z = diB_l_z_batch(l, z, "iB_UWWp", theta_U, theta_T, CosmoClassObject, "B_GM")
        else:
            diB_app_l_z = diB_l_z_batch(l, z, "iB_UWWp", theta_U, theta_T, CosmoClassObject, "B_GMRF")
    integ(diB_app_l_z, nitn=5, neval=2e4) # warm-up the MC grid importance sampling for initial adapting of the grid
    iB_app_l_z = integ(diB_app_l_z, nitn=5, neval=6e5).mean

    return iB_app_l_z  #* (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)


def A2pt_phi_qag_integrand(double phi, void *params)
{
    params_A2pt_phi_integrand *p = static_cast<params_A2pt_phi_integrand *>(params);

    double length = l_ApB(p->theta, phi, p->alpha, p->phi_alpha);

    return W2D_TH_RS_unnormalised(length, p->theta_T);
}

double A2pt_theta_qag_integrand(double theta, void *params)
{
    params_A2pt_theta_phi_integrand *p = static_cast<params_A2pt_theta_phi_integrand *>(params);

    //parameters in integrand
    params_A2pt_phi_integrand args = {theta, p->phi_alpha, p->alpha, p->theta_T};

    double result = 0;              // the result from the integration
    double error = 0;               // the estimated error from the integration

    qag_1D_integration(&A2pt_phi_qag_integrand, static_cast<void *>(&args), 0, 2*M_PI, calls_1e5, result, error);

    return theta*W2D_TH_RS_unnormalised(theta, p->theta_T)*result;
}

double A2pt_qag(double alpha, double theta_T)
{
    double phi_alpha = 0;           // assuming angular-independency we can just fix the phi_alpha to any angle we want
    double result = 0;              // the result from the integration
    double error = 0;               // the estimated error from the integration

    //parameters in integrand
    params_A2pt_theta_phi_integrand args = {phi_alpha, alpha, theta_T};

    qag_1D_integration(&A2pt_theta_qag_integrand, static_cast<void *>(&args), 0, 4.0*theta_T, calls_1e5, result, error);

    return result;
}

'''