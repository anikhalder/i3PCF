import numpy as np
import vegas
import input
from window_utils import W2D_TH_FS, W2D_U_FS
from cosmology_utils import T17_shell_correction
import constants

########################
#### Power spectrum ####
########################

# Function to compute the P(l, z) at a given (l, z) grid point 
def P_l_z(params):
    l = params[0]
    z = params[1]
    CosmoClassObject = params[2]
    P3D_type = params[3]

    k = l/CosmoClassObject.chi_z(z)
    
    if (P3D_type == 'lin'):
        P_l_z_vals = CosmoClassObject.P3D_k_z(k, z, False)
    elif (P3D_type == 'nl'):
        P_l_z_vals = CosmoClassObject.P3D_k_z(k, z, True)

    if (constants._apply_T17_corrections_ == True):
        P_l_z_vals = P_l_z_vals*T17_shell_correction(k/CosmoClassObject.h)**2

    return P_l_z_vals

# Function to compute the scale-dependent stochasticity P_eps_eps(l, z) at a given (l, z) grid point (without 1/n_bar)
def P_eps_eps_l_z(params):
    l = params[0]
    z = params[1]
    CosmoClassObject = params[2]

    k = l/CosmoClassObject.chi_z(z)

    c = -1
    R = 2

    P_eps_eps_l_z_vals = 1 - c*np.exp(-k*k*R*R)

    if (constants._apply_T17_corrections_ == True):
        P_eps_eps_l_z_vals = P_eps_eps_l_z_vals*T17_shell_correction(k/CosmoClassObject.h)**2

    return P_eps_eps_l_z_vals

####################
#### Bispectrum ####
####################

# Function to compute the B(l1, l2, l3, z) at a given (l1, l2, l3, z) grid point 
def B_l1_l2_l3_z(params):
    l1 = params[0]
    l2 = params[1]
    l3 = params[2]
    z = params[3]
    CosmoClassObject = params[4]
    B3D_type = params[5]

    DA_inv = 1./CosmoClassObject.chi_z(z)

    k1 = np.array([l1*DA_inv])
    k2 = np.array([l2*DA_inv])
    k3 = np.array([l3*DA_inv])
    
    if (B3D_type == 'lin'):
        B_l1_l2_l3_z_vals = CosmoClassObject.B3D_k1_k2_k3_z(k1, k2, k3, z, 'B_tree')
    elif (B3D_type == 'nl'):
        B_l1_l2_l3_z_vals = CosmoClassObject.B3D_k1_k2_k3_z(k1, k2, k3, z, 'B_GM')

    if (constants._apply_T17_corrections_ == True):
        B_l1_l2_l3_z_vals = B_l1_l2_l3_z_vals*T17_shell_correction(k1/CosmoClassObject.h)*T17_shell_correction(k2/CosmoClassObject.h)*T17_shell_correction(k3/CosmoClassObject.h)

    return B_l1_l2_l3_z_vals

###############################
#### Integrated bispectrum ####
###############################

#### build a class for the integrated bispectrum integrand for vegas batch integration mode #### 
class diB_l_z_batch(vegas.BatchIntegrand):
    def __init__(self, l, z, iB_type, theta_U, theta_T, CosmoClassObject, B3D_type):
        self.l = l
        self.z = z
        self.iB_type = iB_type
        self.theta_U = theta_U
        self.theta_T = theta_T
        self.CosmoClassObject = CosmoClassObject
        self.B3D_type = B3D_type
        self.DA_inv = 1./CosmoClassObject.chi_z(z)
        self.h = CosmoClassObject.h
    
    def __call__(self, p):
        # evaluate integrand at multiple points simultaneously

        k1 = p[:,0]*self.DA_inv
        k2 = p[:,1]*self.DA_inv
        #k3 = l_ApB(p[:,0],p[:,2],p[:,1],p[:,3])*self.DA_inv
        k3 = np.sqrt(p[:,0]*p[:,0] + p[:,1]*p[:,1] + 2*p[:,0]*p[:,1]*np.cos(p[:,3]-p[:,2]))*self.DA_inv

        #l2pl = l_ApB(p[:,1],p[:,3],self.l,0.0)
        #ml1ml2ml = l_ApBpC(p[:,0],np.pi+p[:,2],p[:,1],np.pi+p[:,3],self.l,np.pi)
        #phi1p2 = phi_ApB(p[:,0],p[:,2],p[:,1],p[:,3])

        l2pl = np.sqrt(p[:,1]*p[:,1] + self.l*self.l + 2*p[:,1]*self.l*np.cos(p[:,3]))
        ml1ml2ml = np.sqrt(self.l*self.l + p[:,0]*p[:,0] + p[:,1]*p[:,1] + 2*self.l*(p[:,0]*np.cos(p[:,2]) + p[:,1]*np.cos(p[:,3])) + 2*p[:,0]*p[:,1]*np.cos(p[:,3]-p[:,2]))

        if (self.iB_type == "iB_WWW"): # for galaxy-galaxy-galaxy i3PCF (or for any scalar field e.g. convergence)
            window_factor = W2D_TH_FS(p[:,0], self.theta_T)*W2D_TH_FS(l2pl, self.theta_T)*W2D_TH_FS(ml1ml2ml, self.theta_T)
        elif (self.iB_type == "iB_UWW"): # for aperture-galaxy-galaxy i3PCF (or for any scalar field e.g. convergence)
            window_factor = W2D_U_FS(p[:,0], self.theta_U)*W2D_TH_FS(l2pl, self.theta_T)*W2D_TH_FS(ml1ml2ml, self.theta_T)
        else:
            y_12 = p[:,0]*np.sin(p[:,2]) + p[:,1]*np.sin(p[:,3])
            x_12 = p[:,0]*np.cos(p[:,2]) + p[:,1]*np.cos(p[:,3])
            phi1p2 = np.arctan2(y_12, x_12)
            mask_12 = (phi1p2 < 0)
            phi1p2[mask_12] =  phi1p2[mask_12] + 2*np.pi   

            if (self.iB_type == "iB_WWWt"): # for galaxy-tangential-shear i3PCF
                window_factor = W2D_TH_FS(p[:,0], self.theta_T)*W2D_TH_FS(l2pl, self.theta_T)*W2D_TH_FS(ml1ml2ml, self.theta_T)*np.cos(2*phi1p2) 
            elif (self.iB_type == "iB_UWWt"): # for aperture-tangential-shear i3PCF
                window_factor = W2D_U_FS(p[:,0], self.theta_U)*W2D_TH_FS(l2pl, self.theta_T)*W2D_TH_FS(ml1ml2ml, self.theta_T)*np.cos(2*phi1p2) 
            elif (self.iB_type == "iB_WWWp"): # for galaxy-shear_p-shear_p  i3PCF+
                window_factor = W2D_TH_FS(p[:,0], self.theta_T)*W2D_TH_FS(l2pl, self.theta_T)*W2D_TH_FS(ml1ml2ml, self.theta_T)*np.cos(2*(p[:,3] - phi1p2))
            elif (self.iB_type == "iB_WWWm"): # for galaxy-shear_m-shear_m i3PCF-
                window_factor = W2D_TH_FS(p[:,0], self.theta_T)*W2D_TH_FS(l2pl, self.theta_T)*W2D_TH_FS(ml1ml2ml, self.theta_T)*np.cos(2*(p[:,3] + phi1p2))
            elif (self.iB_type == "iB_UWWp"): # for aperture-shear_p-shear_p i3PCF+
                window_factor = W2D_U_FS(p[:,0], self.theta_U)*W2D_TH_FS(l2pl, self.theta_T)*W2D_TH_FS(ml1ml2ml, self.theta_T)*np.cos(2*(p[:,3] - phi1p2)) 
            elif (self.iB_type == "iB_UWWm"): # for aperture-shear_m-shear_m i3PCF-
                window_factor = W2D_U_FS(p[:,0], self.theta_U)*W2D_TH_FS(l2pl, self.theta_T)*W2D_TH_FS(ml1ml2ml, self.theta_T)*np.cos(2*(p[:,3] + phi1p2))
            else:
                window_factor = 1.0

        #diB_l_z = np.zeros(k1.size)
        diB_l_z = p[:,0]*p[:,1]*self.CosmoClassObject.B3D_k1_k2_k3_z(k1, k2, k3, self.z, self.B3D_type)*window_factor
        
        if (constants._apply_T17_corrections_ == True):
            diB_l_z = diB_l_z*T17_shell_correction(k1/self.h)*T17_shell_correction(k2/self.h)*T17_shell_correction(k3/self.h)

        return diB_l_z

def iB_app_l_z_integration(params):
    # Function to compute the iB(l,z) at a given (l,z) grid point with vegas integration
    # Wse polar coordinates here where p[2] and p[3] are polar angles (radian) of 2D wavevctors l1 and l2.
    # Without loss of generality can always assume that vector l coincides with the positive x-axis i.e. phi_l = 0.
    l = params[0]
    z = params[1]
    theta_U = params[2]
    theta_T = params[3]
    CosmoClassObject = params[4]
    B3D_type = params[5]

    integ = vegas.Integrator([[0, 20000], [0, 20000], [0, 2*np.pi], [0, 2*np.pi]]) # l1, l2, phi_1, phi_2

    if (B3D_type == 'lin'):
        diB_app_l_z = diB_l_z_batch(l, z, "iB_UWWp", theta_U, theta_T, CosmoClassObject, "B_tree")
    elif (B3D_type == 'nl'):
        if (input.iB_app_type == 'GM'):
            diB_app_l_z = diB_l_z_batch(l, z, "iB_UWWp", theta_U, theta_T, CosmoClassObject, "B_GM")
        else:
            diB_app_l_z = diB_l_z_batch(l, z, "iB_UWWp", theta_U, theta_T, CosmoClassObject, "B_GMRF")
    integ(diB_app_l_z, nitn=5, neval=3e4) # warm-up the MC grid importance sampling for initial adapting of the grid
    iB_app_l_z = integ(diB_app_l_z, nitn=5, neval=7e5).mean

    # settings used previously for 100 ell values
    #integ(diB_app_l_z, nitn=5, neval=2e4) # warm-up the MC grid importance sampling for initial adapting of the grid
    #iB_app_l_z = integ(diB_app_l_z, nitn=5, neval=6e5).mean

    return iB_app_l_z  #* (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)

def iB_amm_l_z_integration(params):  
    l = params[0]
    z = params[1]
    theta_U = params[2]
    theta_T = params[3]
    CosmoClassObject = params[4]
    B3D_type = params[5]

    integ = vegas.Integrator([[0, 20000], [0, 20000], [0, 2*np.pi], [0, 2*np.pi]]) # l1, l2, phi_1, phi_2

    if (B3D_type == 'lin'):
        diB_amm_l_z = diB_l_z_batch(l, z, "iB_UWWm", theta_U, theta_T, CosmoClassObject, "B_tree")
    elif (B3D_type == 'nl'):
        if (input.iB_amm_type == 'GM'):
            diB_amm_l_z = diB_l_z_batch(l, z, "iB_UWWm", theta_U, theta_T, CosmoClassObject, "B_GM")
        else:
            diB_amm_l_z = diB_l_z_batch(l, z, "iB_UWWm", theta_U, theta_T, CosmoClassObject, "B_GMRF")
    integ(diB_amm_l_z, nitn=5, neval=3e4) # warm-up the MC grid
    iB_amm_l_z = integ(diB_amm_l_z, nitn=5, neval=7e5).mean

    return iB_amm_l_z  #* (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)

def iB_att_l_z_integration(params):
    l = params[0]
    z = params[1]
    theta_U = params[2]
    theta_T = params[3]
    CosmoClassObject = params[4]
    B3D_type = params[5]

    integ = vegas.Integrator([[0, 20000], [0, 20000], [0, 2*np.pi], [0, 2*np.pi]]) # l1, l2, phi_1, phi_2

    if (B3D_type == 'lin'):
        diB_att_b1_l_z = diB_l_z_batch(l, z, "iB_UWWt", theta_U, theta_T, CosmoClassObject, "B_tree")
    elif (B3D_type == 'nl'):
        if (input.iB_att_type == 'GM'):
            diB_att_b1_l_z = diB_l_z_batch(l, z, "iB_UWWt", theta_U, theta_T, CosmoClassObject, "B_GM")
        elif (input.iB_att_type == 'GMRF'):
            diB_att_b1_l_z = diB_l_z_batch(l, z, "iB_UWWt", theta_U, theta_T, CosmoClassObject, "B_GMRF")
    integ(diB_att_b1_l_z, nitn=5, neval=3e4) # warm-up the MC grid
    iB_att_b1_l_z = integ(diB_att_b1_l_z, nitn=5, neval=7e5).mean

    if (B3D_type == 'lin'):
        diB_att_b2_l_z = diB_l_z_batch(l, z, "iB_UWWt", theta_U, theta_T, CosmoClassObject, "B_mgm_b2_lin")
    elif (B3D_type == 'nl'):
        diB_att_b2_l_z = diB_l_z_batch(l, z, "iB_UWWt", theta_U, theta_T, CosmoClassObject, "B_mgm_b2_nl")
    integ(diB_att_b2_l_z, nitn=5, neval=3e4) # warm-up the MC grid
    iB_att_b2_l_z = integ(diB_att_b2_l_z, nitn=5, neval=7e5).mean

    if (B3D_type == 'lin'):
        diB_att_bs2_l_z = diB_l_z_batch(l, z, "iB_UWWt", theta_U, theta_T, CosmoClassObject, "B_mgm_bs2_lin")
    elif (B3D_type == 'nl'):
        diB_att_bs2_l_z = diB_l_z_batch(l, z, "iB_UWWt", theta_U, theta_T, CosmoClassObject, "B_mgm_bs2_nl")
    integ(diB_att_bs2_l_z, nitn=5, neval=3e4) # warm-up the MC grid
    iB_att_bs2_l_z = integ(diB_att_bs2_l_z, nitn=5, neval=7e5).mean

    return iB_att_b1_l_z, iB_att_b2_l_z, iB_att_bs2_l_z  #* (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)

def iB_agg_l_z_integration(params):
    l = params[0]
    z = params[1]
    theta_U = params[2]
    theta_T = params[3]
    CosmoClassObject = params[4]
    B3D_type = params[5]

    integ = vegas.Integrator([[0, 20000], [0, 20000], [0, 2*np.pi], [0, 2*np.pi]]) # l1, l2, phi_1, phi_2

    if (B3D_type == 'lin'):
        diB_agg_b1sq_l_z = diB_l_z_batch(l, z, "iB_UWW", theta_U, theta_T, CosmoClassObject, "B_tree")
    elif (B3D_type == 'nl'):
        if (input.iB_agg_type == 'GM'):
            diB_agg_b1sq_l_z = diB_l_z_batch(l, z, "iB_UWW", theta_U, theta_T, CosmoClassObject, "B_GM")
        elif (input.iB_agg_type == 'GMRF'):
            diB_agg_b1sq_l_z = diB_l_z_batch(l, z, "iB_UWW", theta_U, theta_T, CosmoClassObject, "B_GMRF")
    integ(diB_agg_b1sq_l_z, nitn=5, neval=3e4) # warm-up the MC grid
    iB_agg_b1sq_l_z = integ(diB_agg_b1sq_l_z, nitn=5, neval=7e5).mean

    if (B3D_type == 'lin'):
        diB_agg_b1_b2_l_z = diB_l_z_batch(l, z, "iB_UWW", theta_U, theta_T, CosmoClassObject, "B_mgg_b1_b2_lin")
    elif (B3D_type == 'nl'):
        diB_agg_b1_b2_l_z = diB_l_z_batch(l, z, "iB_UWW", theta_U, theta_T, CosmoClassObject, "B_mgg_b1_b2_nl")
    integ(diB_agg_b1_b2_l_z, nitn=5, neval=3e4) # warm-up the MC grid
    iB_agg_b1_b2_l_z = integ(diB_agg_b1_b2_l_z, nitn=5, neval=7e5).mean

    if (B3D_type == 'lin'):
        diB_agg_b1_bs2_l_z = diB_l_z_batch(l, z, "iB_UWW", theta_U, theta_T, CosmoClassObject, "B_mgg_b1_bs2_lin")
    elif (B3D_type == 'nl'):
        diB_agg_b1_bs2_l_z = diB_l_z_batch(l, z, "iB_UWW", theta_U, theta_T, CosmoClassObject, "B_mgg_b1_bs2_nl")
    integ(diB_agg_b1_bs2_l_z, nitn=5, neval=3e4) # warm-up the MC grid
    iB_agg_b1_bs2_l_z = integ(diB_agg_b1_bs2_l_z, nitn=5, neval=7e5).mean

    if (B3D_type == 'lin'):
        diB_agg_eps_epsdelta_l_z = diB_l_z_batch(l, z, "iB_UWW", theta_U, theta_T, CosmoClassObject, "B_mgg_eps_epsdelta_lin")
    elif (B3D_type == 'nl'):
        diB_agg_eps_epsdelta_l_z = diB_l_z_batch(l, z, "iB_UWW", theta_U, theta_T, CosmoClassObject, "B_mgg_eps_epsdelta_nl")
    integ(diB_agg_eps_epsdelta_l_z, nitn=5, neval=3e4) # warm-up the MC grid
    iB_agg_eps_epsdelta_l_z = integ(diB_agg_eps_epsdelta_l_z, nitn=5, neval=7e5).mean

    return iB_agg_b1sq_l_z, iB_agg_b1_b2_l_z, iB_agg_b1_bs2_l_z, iB_agg_eps_epsdelta_l_z  #* (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)

def iB_gpp_l_z_integration(params):
    l = params[0]
    z = params[1]
    theta_U = params[2]
    theta_T = params[3]
    CosmoClassObject = params[4]
    B3D_type = params[5]

    integ = vegas.Integrator([[0, 20000], [0, 20000], [0, 2*np.pi], [0, 2*np.pi]]) # l1, l2, phi_1, phi_2

    if (B3D_type == 'lin'):
        diB_gpp_b1_l_z = diB_l_z_batch(l, z, "iB_WWWp", theta_T, theta_T, CosmoClassObject, "B_tree")
    elif (B3D_type == 'nl'):
        if (input.iB_gpp_type == 'GM'):
            diB_gpp_b1_l_z = diB_l_z_batch(l, z, "iB_WWWp", theta_T, theta_T, CosmoClassObject, "B_GM")
        elif (input.iB_gpp_type == 'GMRF'):
            diB_gpp_b1_l_z = diB_l_z_batch(l, z, "iB_WWWp", theta_T, theta_T, CosmoClassObject, "B_GMRF")
    integ(diB_gpp_b1_l_z, nitn=5, neval=3e4) # warm-up the MC grid importance sampling for initial adapting of the grid
    iB_gpp_b1_l_z = integ(diB_gpp_b1_l_z, nitn=5, neval=7e5).mean

    if (B3D_type == 'lin'):
        diB_gpp_b2_l_z = diB_l_z_batch(l, z, "iB_WWWp", theta_T, theta_T, CosmoClassObject, "B_gmm_b2_lin")
    elif (B3D_type == 'nl'):
        diB_gpp_b2_l_z = diB_l_z_batch(l, z, "iB_WWWp", theta_T, theta_T, CosmoClassObject, "B_gmm_b2_nl")
    integ(diB_gpp_b2_l_z, nitn=5, neval=3e4) # warm-up the MC grid 
    iB_gpp_b2_l_z = integ(diB_gpp_b2_l_z, nitn=5, neval=7e5).mean

    if (B3D_type == 'lin'):
        diB_gpp_bs2_l_z = diB_l_z_batch(l, z, "iB_WWWp", theta_T, theta_T, CosmoClassObject, "B_gmm_bs2_lin")
    elif (B3D_type == 'nl'):
        diB_gpp_bs2_l_z = diB_l_z_batch(l, z, "iB_WWWp", theta_T, theta_T, CosmoClassObject, "B_gmm_bs2_nl")
    integ(diB_gpp_bs2_l_z, nitn=5, neval=3e4) # warm-up the MC grid 
    iB_gpp_bs2_l_z = integ(diB_gpp_bs2_l_z, nitn=5, neval=7e5).mean

    return iB_gpp_b1_l_z, iB_gpp_b2_l_z, iB_gpp_bs2_l_z

def iB_gmm_l_z_integration(params):
    l = params[0]
    z = params[1]
    theta_U = params[2]
    theta_T = params[3]
    CosmoClassObject = params[4]
    B3D_type = params[5]

    integ = vegas.Integrator([[0, 20000], [0, 20000], [0, 2*np.pi], [0, 2*np.pi]]) # l1, l2, phi_1, phi_2

    if (B3D_type == 'lin'):
        diB_gmm_b1_l_z = diB_l_z_batch(l, z, "iB_WWWm", theta_T, theta_T, CosmoClassObject, "B_tree")
    elif (B3D_type == 'nl'):
        if (input.iB_gmm_type == 'GM'):
            diB_gmm_b1_l_z = diB_l_z_batch(l, z, "iB_WWWm", theta_T, theta_T, CosmoClassObject, "B_GM")
        elif (input.iB_gmm_type == 'GMRF'):
            diB_gmm_b1_l_z = diB_l_z_batch(l, z, "iB_WWWm", theta_T, theta_T, CosmoClassObject, "B_GMRF")
    integ(diB_gmm_b1_l_z, nitn=5, neval=3e4) # warm-up the MC grid 
    iB_gmm_b1_l_z = integ(diB_gmm_b1_l_z, nitn=5, neval=7e5).mean

    if (B3D_type == 'lin'):
        diB_gmm_b2_l_z = diB_l_z_batch(l, z, "iB_WWWm", theta_T, theta_T, CosmoClassObject, "B_gmm_b2_lin")
    elif (B3D_type == 'nl'):
        diB_gmm_b2_l_z = diB_l_z_batch(l, z, "iB_WWWm", theta_T, theta_T, CosmoClassObject, "B_gmm_b2_nl")
    integ(diB_gmm_b2_l_z, nitn=5, neval=3e4) # warm-up the MC grid 
    iB_gmm_b2_l_z = integ(diB_gmm_b2_l_z, nitn=5, neval=7e5).mean

    if (B3D_type == 'lin'):
        diB_gmm_bs2_l_z = diB_l_z_batch(l, z, "iB_WWWm", theta_T, theta_T, CosmoClassObject, "B_gmm_bs2_lin")
    elif (B3D_type == 'nl'):
        diB_gmm_bs2_l_z = diB_l_z_batch(l, z, "iB_WWWm", theta_T, theta_T, CosmoClassObject, "B_gmm_bs2_nl")
    integ(diB_gmm_bs2_l_z, nitn=5, neval=3e4) # warm-up the MC grid 
    iB_gmm_bs2_l_z = integ(diB_gmm_bs2_l_z, nitn=5, neval=7e5).mean

    return iB_gmm_b1_l_z, iB_gmm_b2_l_z, iB_gmm_bs2_l_z

def iB_gtt_l_z_integration(params):
    l = params[0]
    z = params[1]
    theta_U = params[2]
    theta_T = params[3]
    CosmoClassObject = params[4]
    B3D_type = params[5]

    integ = vegas.Integrator([[0, 20000], [0, 20000], [0, 2*np.pi], [0, 2*np.pi]]) # l1, l2, phi_1, phi_2

    if (B3D_type == 'lin'):
        diB_gtt_b1sq_l_z = diB_l_z_batch(l, z, "iB_WWWt", theta_T, theta_T, CosmoClassObject, "B_tree")
    elif (B3D_type == 'nl'):
        if (input.iB_gtt_type == 'GM'):
            diB_gtt_b1sq_l_z = diB_l_z_batch(l, z, "iB_WWWt", theta_T, theta_T, CosmoClassObject, "B_GM")
        elif (input.iB_gtt_type == 'GMRF'):
            diB_gtt_b1sq_l_z = diB_l_z_batch(l, z, "iB_WWWt", theta_T, theta_T, CosmoClassObject, "B_GMRF")
    integ(diB_gtt_b1sq_l_z, nitn=5, neval=3e4) # warm-up the MC grid
    iB_gtt_b1sq_l_z = integ(diB_gtt_b1sq_l_z, nitn=5, neval=7e5).mean

    if (B3D_type == 'lin'):
        diB_gtt_b1_b2_l_z = diB_l_z_batch(l, z, "iB_WWWt", theta_T, theta_T, CosmoClassObject, "B_ggm_b1_b2_lin")
    elif (B3D_type == 'nl'):
        diB_gtt_b1_b2_l_z = diB_l_z_batch(l, z, "iB_WWWt", theta_T, theta_T, CosmoClassObject, "B_ggm_b1_b2_nl")
    integ(diB_gtt_b1_b2_l_z, nitn=5, neval=3e4) # warm-up the MC grid
    iB_gtt_b1_b2_l_z = integ(diB_gtt_b1_b2_l_z, nitn=5, neval=7e5).mean

    if (B3D_type == 'lin'):
        diB_gtt_b1_bs2_l_z = diB_l_z_batch(l, z, "iB_WWWt", theta_T, theta_T, CosmoClassObject, "B_ggm_b1_bs2_lin")
    elif (B3D_type == 'nl'):
        diB_gtt_b1_bs2_l_z = diB_l_z_batch(l, z, "iB_WWWt", theta_T, theta_T, CosmoClassObject, "B_ggm_b1_bs2_nl")
    integ(diB_gtt_b1_bs2_l_z, nitn=5, neval=3e4) # warm-up the MC grid
    iB_gtt_b1_bs2_l_z = integ(diB_gtt_b1_bs2_l_z, nitn=5, neval=7e5).mean

    if (B3D_type == 'lin'):
        diB_gtt_eps_epsdelta_l_z = diB_l_z_batch(l, z, "iB_WWWt", theta_T, theta_T, CosmoClassObject, "B_ggm_eps_epsdelta_lin")
    elif (B3D_type == 'nl'):
        diB_gtt_eps_epsdelta_l_z = diB_l_z_batch(l, z, "iB_WWWt", theta_T, theta_T, CosmoClassObject, "B_ggm_eps_epsdelta_nl")
    integ(diB_gtt_eps_epsdelta_l_z, nitn=5, neval=3e4) # warm-up the MC grid
    iB_gtt_eps_epsdelta_l_z = integ(diB_gtt_eps_epsdelta_l_z, nitn=5, neval=7e5).mean

    return iB_gtt_b1sq_l_z, iB_gtt_b1_b2_l_z, iB_gtt_b1_bs2_l_z, iB_gtt_eps_epsdelta_l_z  #* (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)

def iB_ggg_l_z_integration(params):
    l = params[0]
    z = params[1]
    theta_U = params[2]
    theta_T = params[3]
    CosmoClassObject = params[4]
    B3D_type = params[5]

    integ = vegas.Integrator([[0, 20000], [0, 20000], [0, 2*np.pi], [0, 2*np.pi]]) # l1, l2, phi_1, phi_2

    if (B3D_type == 'lin'):
        diB_ggg_b1cu_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_tree")
    elif (B3D_type == 'nl'):
        if (input.iB_ggg_type == 'GM'):
            diB_ggg_b1cu_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_GM")
        elif (input.iB_ggg_type == 'GMRF'):
            diB_ggg_b1cu_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_GMRF")        
    integ(diB_ggg_b1cu_l_z, nitn=5, neval=3e4) # warm-up the MC grid
    iB_ggg_b1cu_l_z = integ(diB_ggg_b1cu_l_z, nitn=5, neval=7e5).mean

    if (B3D_type == 'lin'):
        diB_ggg_b1sq_b2_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_ggg_b1sq_b2_lin")
    elif (B3D_type == 'nl'):
        diB_ggg_b1sq_b2_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_ggg_b1sq_b2_nl")
    integ(diB_ggg_b1sq_b2_l_z, nitn=5, neval=3e4) # warm-up the MC grid
    iB_ggg_b1sq_b2_l_z = integ(diB_ggg_b1sq_b2_l_z, nitn=5, neval=7e5).mean
    
    if (B3D_type == 'lin'):
        diB_ggg_b1sq_bs2_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_ggg_b1sq_bs2_lin")
    elif (B3D_type == 'nl'):
        diB_ggg_b1sq_bs2_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_ggg_b1sq_bs2_nl")
    integ(diB_ggg_b1sq_bs2_l_z, nitn=5, neval=3e4) # warm-up the MC grid
    iB_ggg_b1sq_bs2_l_z = integ(diB_ggg_b1sq_bs2_l_z, nitn=5, neval=7e5).mean

    if (B3D_type == 'lin'):
        diB_ggg_b1_eps_epsdelta_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_ggg_b1_eps_epsdelta_lin")
    elif (B3D_type == 'nl'):
        diB_ggg_b1_eps_epsdelta_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_ggg_b1_eps_epsdelta_nl")
    integ(diB_ggg_b1_eps_epsdelta_l_z, nitn=5, neval=3e4) # warm-up the MC grid
    iB_ggg_b1_eps_epsdelta_l_z = integ(diB_ggg_b1_eps_epsdelta_l_z, nitn=5, neval=7e5).mean

    # careful with this as it can return flat spectrum leading to troubles in real space (better to calculate analytcally?)
    diB_ggg_eps_eps_eps_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_ggg_eps_eps_eps")
    integ(diB_ggg_eps_eps_eps_l_z, nitn=5, neval=3e4) # warm-up the MC grid
    iB_ggg_eps_eps_eps_l_z = integ(diB_ggg_eps_eps_eps_l_z, nitn=5, neval=7e5).mean

    return iB_ggg_b1cu_l_z, iB_ggg_b1sq_b2_l_z, iB_ggg_b1sq_bs2_l_z, iB_ggg_b1_eps_epsdelta_l_z, iB_ggg_eps_eps_eps_l_z  #* (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)

def iB_akk_l_z_integration(params):
    # Function to compute the iB(l,z) at a given (l,z) grid point with vegas integration
    # Wse polar coordinates here where p[2] and p[3] are polar angles (radian) of 2D wavevctors l1 and l2.
    # Without loss of generality can always assume that vector l coincides with the positive x-axis i.e. phi_l = 0.
    l = params[0]
    z = params[1]
    theta_U = params[2]
    theta_T = params[3]
    CosmoClassObject = params[4]
    B3D_type = params[5]

    integ = vegas.Integrator([[0, 20000], [0, 20000], [0, 2*np.pi], [0, 2*np.pi]]) # l1, l2, phi_1, phi_2

    if (B3D_type == 'lin'):
        diB_akk_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_tree")
    elif (B3D_type == 'nl'):
        diB_akk_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_GMRF")
    integ(diB_akk_l_z, nitn=5, neval=3e4) # warm-up the MC grid importance sampling for initial adapting of the grid
    iB_akk_l_z = integ(diB_akk_l_z, nitn=5, neval=7e5).mean

    return iB_akk_l_z  #* (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)

def iB_agk_l_z_integration(params):
    l = params[0]
    z = params[1]
    theta_U = params[2]
    theta_T = params[3]
    CosmoClassObject = params[4]
    B3D_type = params[5]

    integ = vegas.Integrator([[0, 20000], [0, 20000], [0, 2*np.pi], [0, 2*np.pi]]) # l1, l2, phi_1, phi_2

    if (B3D_type == 'lin'):
        diB_agk_b1_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_tree")
    elif (B3D_type == 'nl'):
        diB_agk_b1_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_GMRF")
    integ(diB_agk_b1_l_z, nitn=5, neval=3e4) # warm-up the MC grid
    iB_agk_b1_l_z = integ(diB_agk_b1_l_z, nitn=5, neval=7e5).mean

    if (B3D_type == 'lin'):
        diB_agk_b2_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_mgm_b2_lin")
    elif (B3D_type == 'nl'):
        diB_agk_b2_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_mgm_b2_nl")
    integ(diB_agk_b2_l_z, nitn=5, neval=3e4) # warm-up the MC grid
    iB_agk_b2_l_z = integ(diB_agk_b2_l_z, nitn=5, neval=7e5).mean

    if (B3D_type == 'lin'):
        diB_agk_bs2_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_mgm_bs2_lin")
    elif (B3D_type == 'nl'):
        diB_agk_bs2_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_mgm_bs2_nl")
    integ(diB_agk_bs2_l_z, nitn=5, neval=3e4) # warm-up the MC grid
    iB_agk_bs2_l_z = integ(diB_agk_bs2_l_z, nitn=5, neval=7e5).mean

    return iB_agk_b1_l_z, iB_agk_b2_l_z, iB_agk_bs2_l_z  #* (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)

def iB_gkk_l_z_integration(params):
    l = params[0]
    z = params[1]
    theta_U = params[2]
    theta_T = params[3]
    CosmoClassObject = params[4]
    B3D_type = params[5]

    integ = vegas.Integrator([[0, 20000], [0, 20000], [0, 2*np.pi], [0, 2*np.pi]]) # l1, l2, phi_1, phi_2

    if (B3D_type == 'lin'):
        diB_gkk_b1_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_tree")
    elif (B3D_type == 'nl'):
        diB_gkk_b1_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_GMRF")
    integ(diB_gkk_b1_l_z, nitn=5, neval=3e4) # warm-up the MC grid importance sampling for initial adapting of the grid
    iB_gkk_b1_l_z = integ(diB_gkk_b1_l_z, nitn=5, neval=7e5).mean

    if (B3D_type == 'lin'):
        diB_gkk_b2_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_gmm_b2_lin")
    elif (B3D_type == 'nl'):
        diB_gkk_b2_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_gmm_b2_nl")
    integ(diB_gkk_b2_l_z, nitn=5, neval=3e4) # warm-up the MC grid 
    iB_gkk_b2_l_z = integ(diB_gkk_b2_l_z, nitn=5, neval=7e5).mean

    if (B3D_type == 'lin'):
        diB_gkk_bs2_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_gmm_bs2_lin")
    elif (B3D_type == 'nl'):
        diB_gkk_bs2_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_gmm_bs2_nl")
    integ(diB_gkk_bs2_l_z, nitn=5, neval=3e4) # warm-up the MC grid 
    iB_gkk_bs2_l_z = integ(diB_gkk_bs2_l_z, nitn=5, neval=7e5).mean

    return iB_gkk_b1_l_z, iB_gkk_b2_l_z, iB_gkk_bs2_l_z

def iB_ggk_l_z_integration(params):
    l = params[0]
    z = params[1]
    theta_U = params[2]
    theta_T = params[3]
    CosmoClassObject = params[4]
    B3D_type = params[5]

    integ = vegas.Integrator([[0, 20000], [0, 20000], [0, 2*np.pi], [0, 2*np.pi]]) # l1, l2, phi_1, phi_2

    if (B3D_type == 'lin'):
        diB_ggk_b1sq_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_tree")
    elif (B3D_type == 'nl'):
        diB_ggk_b1sq_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_GMRF")
    integ(diB_ggk_b1sq_l_z, nitn=5, neval=3e4) # warm-up the MC grid
    iB_ggk_b1sq_l_z = integ(diB_ggk_b1sq_l_z, nitn=5, neval=7e5).mean

    if (B3D_type == 'lin'):
        diB_ggk_b1_b2_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_ggm_b1_b2_lin")
    elif (B3D_type == 'nl'):
        diB_ggk_b1_b2_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_ggm_b1_b2_nl")
    integ(diB_ggk_b1_b2_l_z, nitn=5, neval=3e4) # warm-up the MC grid
    iB_ggk_b1_b2_l_z = integ(diB_ggk_b1_b2_l_z, nitn=5, neval=7e5).mean

    if (B3D_type == 'lin'):
        diB_ggk_b1_bs2_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_ggm_b1_bs2_lin")
    elif (B3D_type == 'nl'):
        diB_ggk_b1_bs2_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_ggm_b1_bs2_nl")
    integ(diB_ggk_b1_bs2_l_z, nitn=5, neval=3e4) # warm-up the MC grid
    iB_ggk_b1_bs2_l_z = integ(diB_ggk_b1_bs2_l_z, nitn=5, neval=7e5).mean

    if (B3D_type == 'lin'):
        diB_ggk_eps_epsdelta_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_ggm_eps_epsdelta_lin")
    elif (B3D_type == 'nl'):
        diB_ggk_eps_epsdelta_l_z = diB_l_z_batch(l, z, "iB_WWW", theta_T, theta_T, CosmoClassObject, "B_ggm_eps_epsdelta_nl")
    integ(diB_ggk_eps_epsdelta_l_z, nitn=5, neval=3e4) # warm-up the MC grid
    iB_ggk_eps_epsdelta_l_z = integ(diB_ggk_eps_epsdelta_l_z, nitn=5, neval=7e5).mean

    return iB_ggk_b1sq_l_z, iB_ggk_b1_b2_l_z, iB_ggk_b1_bs2_l_z, iB_ggk_eps_epsdelta_l_z  #* (np.pi * theta_T**2)**2 * (1./(2.*np.pi)**4)